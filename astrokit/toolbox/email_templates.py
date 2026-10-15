"""Editable astrokit mail text and HTML layout; no account information here."""
from datetime import datetime
from html import escape
from pathlib import Path
import socket
from string import Template
import sys

__all__ = []

# Only the template text is substituted. Literal body text is never interpreted.
TEMPLATES = {
    "notification": "程序已执行到邮件通知步骤。\n\nThe program has reached this notification step.",
    "completed": "任务已完成：$task\n$details\n\nTask completed: $task\n$details",
    "failed": "任务执行失败：$task\n原因：$error\n\nTask failed: $task\nReason: $error",
}

HTML_TEMPLATE = Template("""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;background:#f5f7fa;">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0"><tr><td align="center" style="padding:24px 12px;">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="max-width:640px;background:#fff;border:1px solid #e2e8ef;border-radius:8px;font-family:Arial,'Microsoft YaHei',sans-serif;color:#243447;line-height:1.7;font-size:14px;">
<tr><td style="padding:20px 24px;border-bottom:1px solid #e2e8ef;">
<div style="font-size:23px;font-weight:bold;color:#225a82;">astrokit</div>
<div style="font-size:12px;color:#61758a;">自动通知 / Automated notification</div></td></tr>
<tr><td style="padding:24px;">
<h2 style="margin:0 0 18px;font-size:18px;color:#225a82;">$subject</h2>
<div style="overflow-wrap:anywhere;white-space:pre-wrap;">$body</div>
<hr style="border:0;border-top:1px solid #e2e8ef;margin:24px 0 16px;">
<div style="font-size:12px;color:#61758a;">$footer</div>
</td></tr></table></td></tr></table></body></html>""")


def render_email(subject, body=None, *, template="notification", context=None):
    """Build a branded plain-text/HTML pair with safely escaped user content."""
    subject = str(subject).strip()
    if not subject or "\r" in subject or "\n" in subject:
        raise ValueError("subject must be nonempty and contain no newlines")
    subject = "[astrokit] " + subject.removeprefix("[astrokit]").lstrip()
    if body is None:
        source = TEMPLATES.get(str(template))
        if source is None:
            source = Path(template).expanduser().read_text(encoding="utf-8")
        try:
            body = Template(source).substitute(context or {})
        except (KeyError, ValueError):
            raise ValueError("Invalid email template or missing context variables") from None
    else:
        body = str(body)
    footer = ("astrokit | 自动通知 / Automated notification\n"
              f"Host: {socket.gethostname()}\n"
              f"Program: {Path(sys.argv[0]).name or 'Python'}\n"
              f"Time: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    text = body + "\n\n--\n" + footer
    html = HTML_TEMPLATE.substitute(subject=escape(subject), body=escape(body),
                                    footer=escape(footer).replace("\n", "<br>"))
    return subject, text, html
