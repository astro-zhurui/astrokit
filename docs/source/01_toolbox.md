# astrokit.toolbox

This module provides various convenience functions for various tasks commonly used in my research work. Such as magnitude conversions and some functions related to calculations, cross-matching, and useful functions for plotting.

Any functions in this module can be called via the unified ways:
```python
from astrokit.toolbox import <function_name>
```
For example:
```python
from astrokit.toolbox import fnu_to_ABmag
```

And you can also use the quick access (some commonly used fuctions only show in <_top_level_map.py> file):
```python
import astrokit as ak

tbl = ak.read(<fits_file_path>)
ak.imshow(<image_data>)
```

## Email notifications

Send a notification when a script reaches a chosen step:

```python
import astrokit as ak

ak.send_email("you@example.org", "任务完成", "数据处理已完成，共得到 42 个源。")
```

The subject automatically becomes `[astrokit] 任务完成`. Each email includes
plain-text and HTML versions with astrokit branding, the host, program name,
and timestamp. The call waits for SMTP acceptance and returns the Message-ID;
acceptance does not confirm delivery to the recipient's inbox.

Each user configures their own `~/.astrokit_config.yaml`. Add this section to
the existing file, keeping other settings intact:

```yaml
email:
  smtp_host: smtp.example.org
  smtp_port: 465
  security: ssl                 # ssl or starttls (usually port 587)
  username: your-login@example.org
  password: "YOUR_CLIENT_PASSWORD"
  from_address: your-address@example.org  # optional; defaults to username
  display_name: astrokit
  max_message_mb: 25            # total encoded message, including attachments
```

Use the mail provider's client/app password where required. Protect the file:

```sh
chmod 600 ~/.astrokit_config.yaml
```

The mail function reads this private file on every call and refuses files
owned by another user or readable/writable by group or others. It always
verifies TLS certificates, never logs credentials, and never retries a send
automatically. No credentials belong in scripts, templates, or version control.

Use a built-in template and optionally attach a small result file:

```python
ak.send_email(
    "you@example.org", "处理完成",
    template="completed",
    context={"task": "Catalogue matching", "details": "42 sources processed."},
    attachments="summary.txt",
)

ak.send_email(
    ["one@example.org", "two@example.org"], "任务失败",
    template="failed",
    context={"task": "Catalogue matching", "error": "Input file missing."},
)
```

Templates and the shared HTML layout are in `astrokit/toolbox/email_templates.py`.
`notification` is the default; `completed` needs `task` and `details`; `failed`
needs `task` and `error`. Alternatively, set `template` to a UTF-8 text file
containing `$name` placeholders, with values supplied by `context`. Use `$$`
for a literal dollar sign. Passing `body` uses that literal text instead of a
template. HTML special characters in all body text are escaped automatically.

`attachments` accepts one path or an iterable of paths. The default timeout is
30 seconds per network operation and can be changed with `timeout=60`.

Delivery errors raise `astrokit.toolbox.email.EmailSendError`. For partial
acceptance, its `.accepted` and `.refused` attributes list the corresponding
recipients; retry only `.refused`. Connection loss during submission can leave
delivery unconfirmed, so inspect the mailbox before resending. Invalid config,
template, addresses, or oversized messages fail before submission. SMTP sends
do not create an IMAP Sent-folder copy.
