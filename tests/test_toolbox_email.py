"""Bounded email checks: SMTP is mocked; these tests never send mail."""
from email import policy
from email.parser import BytesParser
import importlib
from pathlib import Path
import smtplib
import ssl
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

mail = importlib.import_module("astrokit.toolbox.email")


class EmailTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.path = self.home / ".astrokit_config.yaml"
        self.settings = {"email": {
            "smtp_host": "smtp.example.org", "username": "login@example.org",
            "password": "test-password-never-log", "from_address": "sender@example.org",
        }}
        self.save_config()
        self.home_patch = patch.object(mail.Path, "home", return_value=self.home)
        self.home_patch.start()
        self.addCleanup(self.home_patch.stop)
        self.client = MagicMock()
        self.client.esmtp_features = {"size": "26214400"}
        self.client.sendmail.return_value = {}
        self.smtp_patch = patch.object(mail.smtplib, "SMTP_SSL", return_value=self.client)
        self.smtp = self.smtp_patch.start()
        self.addCleanup(self.smtp_patch.stop)
        self.plain_patch = patch.object(mail.smtplib, "SMTP", return_value=self.client)
        self.plain = self.plain_patch.start()
        self.addCleanup(self.plain_patch.stop)

    def save_config(self):
        self.path.write_text(yaml.safe_dump(self.settings), encoding="utf-8")
        self.path.chmod(0o600)

    def message(self):
        return BytesParser(policy=policy.default).parsebytes(self.client.sendmail.call_args.args[2])

    def test_public_export_plain_html_and_envelope(self):
        import astrokit as ak
        self.assertIs(ak.send_email, mail.send_email)
        message_id = ak.send_email("target@example.org", "任务完成", "Literal $x {y} <script> & 中文")
        message = self.message()
        self.assertEqual(message_id, message["Message-ID"])
        self.assertEqual(message["Subject"], "[astrokit] 任务完成")
        self.assertEqual(message["From"].addresses[0].addr_spec, "sender@example.org")
        self.assertEqual(self.client.sendmail.call_args.args[:2],
                         ("sender@example.org", ["target@example.org"]))
        self.assertIn("Literal $x {y} <script> & 中文", message.get_body(("plain",)).get_content())
        html = message.get_body(("html",)).get_content()
        self.assertIn("&lt;script&gt; &amp; 中文", html)
        self.assertNotIn("<script>", html)
        self.assertNotIn(self.settings["email"]["password"], message.as_string())
        self.assertEqual(message["Auto-Submitted"], "auto-generated")
        tls = self.smtp.call_args.kwargs["context"]
        self.assertTrue(tls.check_hostname)
        self.assertEqual(tls.verify_mode, ssl.CERT_REQUIRED)

    def test_default_and_builtin_templates(self):
        mail.send_email("target@example.org")
        self.assertIn("notification step", self.message().get_body(("plain",)).get_content())
        mail.send_email("target@example.org", "[astrokit] Finished", template="completed",
                        context={"task": "sample", "details": "42 sources"})
        self.assertEqual(self.message()["Subject"], "[astrokit] Finished")
        self.assertIn("42 sources", self.message().get_body(("plain",)).get_content())
        mail.send_email("target@example.org", template="failed", context={"task": "x", "error": "test"})
        self.assertIn("Task failed: x", self.message().get_body(("plain",)).get_content())

    def test_custom_template_attachment_and_unique_recipients(self):
        template = self.home / "body.txt"
        template.write_text("Completed $task; $$ literal", encoding="utf-8")
        attachment = self.home / "结果.txt"
        attachment.write_text("small result", encoding="utf-8")
        mail.send_email(["one@example.org", "two@example.org", "one@example.org"],
                        template=template, context={"task": "check"}, attachments=attachment)
        self.assertEqual(self.client.sendmail.call_args.args[1], ["one@example.org", "two@example.org"])
        self.assertIn("Completed check; $ literal", self.message().get_body(("plain",)).get_content())
        part = list(self.message().iter_attachments())[0]
        self.assertEqual(part.get_filename(), "结果.txt")
        self.assertEqual(part.get_payload(decode=True), b"small result")

    def test_starttls_before_login(self):
        self.settings["email"]["security"] = "starttls"
        self.save_config()
        mail.send_email("target@example.org")
        self.smtp.assert_not_called()
        self.assertEqual(self.plain.call_args.args[1], 587)
        calls = [call[0] for call in self.client.mock_calls]
        self.assertLess(calls.index("starttls"), calls.index("login"))

    def test_private_configuration_required(self):
        self.path.chmod(0o644)
        with self.assertRaises(PermissionError):
            mail.send_email("target@example.org")
        self.smtp.assert_not_called()

    def test_missing_and_invalid_config_do_not_echo_values(self):
        self.path.write_text("email: [test-password-never-log", encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            mail.send_email("target@example.org")
        self.assertNotIn("test-password-never-log", str(caught.exception))
        self.settings = {"unrelated": True}
        self.save_config()
        with self.assertRaisesRegex(ValueError, "email section"):
            mail.send_email("target@example.org")
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            mail.send_email("target@example.org")
        self.smtp.assert_not_called()

    def test_bad_inputs_fail_before_connection(self):
        for recipient in ([], "bad-address", "a@example.org\nBcc:x@example.org", "A <a@example.org>"):
            with self.subTest(recipient=recipient), self.assertRaises(ValueError):
                mail.send_email(recipient)
        with self.assertRaises(ValueError):
            mail.send_email("target@example.org", "title\nBcc: x@example.org")
        with self.assertRaisesRegex(ValueError, "missing context"):
            mail.send_email("target@example.org", template="completed")
        self.settings["email"]["security"] = "none"
        self.save_config()
        with self.assertRaises(ValueError):
            mail.send_email("target@example.org")
        self.smtp.assert_not_called()

    def test_attachment_size_limit_before_connection(self):
        self.settings["email"]["max_message_mb"] = 0.001
        self.save_config()
        attachment = self.home / "large.bin"
        attachment.write_bytes(b"x" * 2048)
        with self.assertRaisesRegex(ValueError, "Attachments exceed"):
            mail.send_email("target@example.org", attachments=attachment)
        with self.assertRaisesRegex(ValueError, "Encoded email exceeds"):
            mail.send_email("target@example.org")
        self.smtp.assert_not_called()

    def test_auth_failure_is_sanitized_and_not_retried(self):
        self.client.login.side_effect = smtplib.SMTPAuthenticationError(
            535, b"server reflected test-password-never-log")
        with self.assertRaises(mail.EmailSendError) as caught:
            mail.send_email("target@example.org")
        self.assertNotIn("test-password-never-log", str(caught.exception))
        self.assertIn("SMTPAuthenticationError", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)
        self.smtp.assert_called_once()
        self.client.sendmail.assert_not_called()

    def test_partial_delivery_exposes_recipient_outcomes(self):
        self.client.sendmail.return_value = {"bad@example.org": (550, b"no")}
        with self.assertRaises(mail.EmailSendError) as caught:
            mail.send_email(["ok@example.org", "bad@example.org"])
        self.assertEqual(caught.exception.accepted, ("ok@example.org",))
        self.assertEqual(caught.exception.refused, ("bad@example.org",))
        self.client.sendmail.assert_called_once()

    def test_all_recipients_refused(self):
        self.client.sendmail.side_effect = smtplib.SMTPRecipientsRefused({"bad@example.org": (550, b"no")})
        with self.assertRaises(mail.EmailSendError) as caught:
            mail.send_email("bad@example.org")
        self.assertEqual(caught.exception.refused, ("bad@example.org",))
        self.assertEqual(caught.exception.accepted, ())

    def test_quit_failure_does_not_turn_acceptance_into_failure(self):
        self.client.quit.side_effect = smtplib.SMTPServerDisconnected("closed")
        self.assertTrue(mail.send_email("target@example.org"))
        self.client.sendmail.assert_called_once()
        self.client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
