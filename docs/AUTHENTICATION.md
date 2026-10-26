# Browser login

The browser app uses a username (email address) and password. Signup asks for those two values; the display name is derived from the username. Each account receives an isolated project under
`DISCOURSELENS_PROJECT_DIR/tenants/`.

Passwords are stored as salted scrypt hashes in a private SQLite database. The
database never stores the original passwords. Login attempts are rate-limited.

## Start and register

Run the local launcher:

```bash
./scripts/start-local.sh
```

Open `http://127.0.0.1:8000`, choose **Crear una cuenta**, then enter a username
and a password of at least 12 characters. Registration is enabled for this
local app, and the host port is bound to localhost.

The launcher keeps the session key and encrypted-provider key in a private
`.env.local` file. Keep it stable across restarts. Account records and
workspaces live in the project directory; do not commit that directory.

Registration stays open for as long as `DISCOURSELENS_WEB_ALLOW_REGISTRATION`
is `true`, which is the default. That is right for localhost, but when the app
is reachable from outside the machine (reverse proxy, Tailscale Funnel) it lets
anyone who can load the page create an account; see
[Remote access](WEB_DEPLOYMENT.md#remote-access). Five failed sign-ins within
15 minutes lock that sign-in for 15 minutes, and a session lasts 8 hours.

## Manage accounts

The Docker image includes `discourselens-web-user` for an operator to add
accounts. For example:

```bash
docker compose --env-file .env.local -f compose.web.yaml run --rm web \
  discourselens-web-user --project-dir /workspace/project create-user \
  --name "Researcher Name" --email researcher@example.org
```

The command prompts for the password twice. Do not put a real password in a
command line or commit account database files. To inspect the account count,
replace `create-user ...` with `count`.

MCP connector authorization is a separate feature with its own configuration;
it does not reuse the browser account or session. See the [MCP setup guide](MCP_SETUP.md).
