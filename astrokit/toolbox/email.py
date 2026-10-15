"""Send script notifications using the current user's private astrokit config."""
from collections.abc import Mapping
from datetime import datetime
from email.headerregistry import Address
from email.message import EmailMessage
from email.policy import SMTP
from email.utils import format_datetime, make_msgid
import math
import mimetypes
import os
from pathlib import Path
import smtplib
import ssl
import stat
from time import perf_counter

from loguru import logger
import yaml

from .email_templates import render_email

__all__ = ["send_email", "EmailSendError"]


class EmailSendError(RuntimeError):
    """Delivery failed or is unconfirmed; accepted/refused list SMTP outcomes."""

    def __init__(self, message, *, message_id=None, accepted=(), refused=()):
        super().__init__(message)
        self.message_id = message_id
        self.accepted = tuple(accepted)
        self.refused = tuple(refused)


def send_email(to, subject="Notification", body=None, *, template="notification",
               context=None, attachments=None, timeout=30):
    """Send one astrokit notification and return its Message-ID.

    ``to`` is an email address or iterable of addresses. ``body`` is literal
    plain text, also safely escaped into the HTML version. Alternatively, use
    a built-in template (notification/completed/failed) or a UTF-8 text file
    with ``$name`` placeholders and a ``context`` mapping. ``attachments`` is
    a path or iterable of paths. Subjects always start with ``[astrokit]``.

    Read SMTP settings from ``email`` in ``~/.astrokit_config.yaml`` on every
    call. The file must belong to the current user and have permissions 600
    (400 also works). SSL/TLS certificate verification is always enabled.
    Return only after SMTP acceptance; this does not confirm inbox delivery.
    Failures raise EmailSendError and are never retried automatically.
    """
    started = perf_counter()
    config_path = Path.home() / ".astrokit_config.yaml"
    info = config_path.stat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o077):
        raise PermissionError(
            "Email config must be a private file owned by the current user; "
            "run chmod 600 ~/.astrokit_config.yaml")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        raise ValueError("Invalid YAML in ~/.astrokit_config.yaml") from None
    if not isinstance(config, Mapping) or not isinstance(config.get("email"), Mapping):
        raise ValueError("Add an email section to ~/.astrokit_config.yaml; see the config template")
    config = config["email"]
    for key in ("smtp_host", "username", "password"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing or invalid email.{key} in ~/.astrokit_config.yaml")
    security = config.get("security", "ssl")
    if security not in ("ssl", "starttls"):
        raise ValueError("email.security must be ssl or starttls; unencrypted SMTP is disabled")
    port = int(config.get("smtp_port", 465 if security == "ssl" else 587))
    max_bytes = float(config.get("max_message_mb", 25)) * 1024**2
    if not 0 < port < 65536 or not math.isfinite(max_bytes) or max_bytes <= 0:
        raise ValueError("Invalid email.smtp_port or email.max_message_mb")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number of seconds")

    recipients = [to] if isinstance(to, str) else list(to)
    if not recipients:
        raise ValueError("At least one recipient is required")
    sender = config.get("from_address") or config["username"]
    addresses = []
    for value in [sender, *recipients]:
        try:
            if not isinstance(value, str) or any(c.isspace() for c in value):
                raise ValueError
            value.encode("ascii")
            address = Address(addr_spec=value)
            if not address.username or not address.domain:
                raise ValueError
        except (ValueError, TypeError, UnicodeError):
            raise ValueError("Use bare ASCII email addresses such as name@example.org") from None
        addresses.append(str(address))
    sender, recipients = addresses[0], list(dict.fromkeys(addresses[1:]))
    title, text, html = render_email(subject, body, template=template, context=context)
    message = EmailMessage()
    message["From"] = Address(config.get("display_name", "astrokit"), addr_spec=sender)
    message["To"] = ", ".join(recipients)
    message["Subject"] = title
    message["Date"] = format_datetime(datetime.now().astimezone())
    message["Message-ID"] = make_msgid(domain=sender.rsplit("@", 1)[1])
    message["Auto-Submitted"] = "auto-generated"
    message["X-Mailer"] = "astrokit"
    message.set_content(text)
    message.add_alternative(html, subtype="html")
    paths = [attachments] if isinstance(attachments, (str, os.PathLike)) else (attachments or [])
    total_size = 0
    for item in paths:
        path = Path(item).expanduser()
        total_size += path.stat().st_size
        if total_size > max_bytes:
            raise ValueError("Attachments exceed email.max_message_mb")
        mime, encoding = mimetypes.guess_type(path.name)
        maintype, subtype = (mime if mime and not encoding else "application/octet-stream").split("/", 1)
        message.add_attachment(path.read_bytes(), maintype=maintype, subtype=subtype, filename=path.name)
    raw = message.as_bytes(policy=SMTP)
    if len(raw) > max_bytes:
        raise ValueError("Encoded email exceeds email.max_message_mb; reduce attachments or body size")

    client = None
    message_id = str(message["Message-ID"])
    try:
        tls = ssl.create_default_context()
        if security == "ssl":
            client = smtplib.SMTP_SSL(config["smtp_host"], port, timeout=timeout, context=tls)
        else:
            client = smtplib.SMTP(config["smtp_host"], port, timeout=timeout)
            client.ehlo()
            client.starttls(context=tls)
        client.ehlo()
        limit = client.esmtp_features.get("size", "")
        if limit.isdigit() and len(raw) > int(limit):
            raise ValueError("Encoded email exceeds the SMTP server message-size limit")
        client.login(config["username"], config["password"])
        refused = client.sendmail(sender, recipients, raw)
    except smtplib.SMTPRecipientsRefused as error:
        raise EmailSendError("SMTP refused all recipients", message_id=message_id,
                             refused=recipients) from None
    except (OSError, smtplib.SMTPException) as error:
        raise EmailSendError(
            f"Email delivery failed or is unconfirmed ({type(error).__name__}); "
            "check the mailbox before retrying", message_id=message_id) from None
    finally:
        if client is not None:
            try:
                client.quit()
            except (OSError, smtplib.SMTPException):
                client.close()
    if refused:
        accepted = [address for address in recipients if address not in refused]
        raise EmailSendError(
            "Partial email delivery; retry only refused recipients (see .accepted and .refused)",
            message_id=message_id, accepted=accepted, refused=refused)
    logger.info("Email accepted by SMTP | recipients={} | id={} | elapsed={:.1f}s",
                len(recipients), message_id, perf_counter() - started)
    return message_id
