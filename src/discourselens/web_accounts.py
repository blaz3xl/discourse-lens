"""Local password accounts for the browser workspace."""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_EMAIL_LIMIT = 254
_PASSWORD_LIMIT = 1024
_SCRYPT_N = 1 << 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_BYTES = 32
_LOGIN_WINDOW_SECONDS = 15 * 60
_LOGIN_LOCK_SECONDS = 15 * 60
_LOGIN_FAILURE_LIMIT = 5
_WORKSPACE_ID = re.compile(r"^(?:[0-9a-f]{32}|[0-9a-f]{64})$")
_DUMMY_SALT = b"discourselens-login-dummy"
_DUMMY_HASH = hashlib.scrypt(
    b"not-a-real-password",
    salt=_DUMMY_SALT,
    n=_SCRYPT_N,
    r=_SCRYPT_R,
    p=_SCRYPT_P,
    dklen=_SCRYPT_BYTES,
    maxmem=64 * 1024 * 1024,
)


class AccountError(ValueError):
    """Base class for expected local-account errors."""


class AccountAlreadyExists(AccountError):
    """Raised when a local email address is already registered."""


class AccountsAlreadyInitialized(AccountError):
    """Raised when first-account setup has already completed."""


@dataclass(frozen=True)
class LoginResult:
    user: dict[str, str] | None
    rate_limited: bool = False


class WebAccountStore:
    """Store password hashes and opaque workspace IDs in a private SQLite file."""

    def __init__(self, project_root: str | Path) -> None:
        root = Path(project_root).expanduser().resolve()
        self.project_root = root
        metadata_dir = root / ".discourselens"
        if metadata_dir.is_symlink():
            raise AccountError("La carpeta de cuentas no puede ser un enlace simbólico")
        self.database_path = metadata_dir / "web_accounts.sqlite3"

    def initialize(self) -> None:
        project_root = self.project_root.resolve()
        parent = self.database_path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if parent.is_symlink() or self.database_path.is_symlink():
            raise AccountError("La base de datos de cuentas no puede ser un enlace simbólico")
        resolved_parent = parent.resolve()
        database_path = self.database_path.resolve(strict=False)
        if (
            not resolved_parent.is_relative_to(project_root)
            or not database_path.is_relative_to(project_root)
        ):
            raise AccountError("La base de datos de cuentas debe estar dentro del proyecto")
        _restrict_directory_permissions(resolved_parent)
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS web_users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    workspace_id TEXT NOT NULL UNIQUE,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS web_login_attempts (
                    identifier_hash TEXT PRIMARY KEY,
                    failures INTEGER NOT NULL,
                    window_started REAL NOT NULL,
                    blocked_until REAL NOT NULL
                );
                """
            )
        _restrict_database_permissions(database_path)

    def count(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT COUNT(*) FROM web_users").fetchone()
            return int(row[0])

    def get_user(self, user_id: str) -> dict[str, str] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT user_id, email, name, workspace_id, created_at, is_active
                FROM web_users WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        return self._public_user(row) if row else None

    def create_user(
        self,
        *,
        email: str,
        name: str,
        password: str,
        workspace_id: str | None = None,
        first_user_only: bool = False,
    ) -> dict[str, str]:
        normalized_email = _normalize_email(email)
        normalized_name = _normalize_name(name)
        _validate_password(password)
        workspace = workspace_id or secrets.token_hex(16)
        if not _WORKSPACE_ID.fullmatch(workspace):
            raise AccountError("El identificador del espacio de trabajo no es válido")
        password_hash = _hash_password(password)
        user_id = secrets.token_hex(16)
        created_at = datetime.now(UTC).isoformat()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if first_user_only and connection.execute(
                    "SELECT 1 FROM web_users LIMIT 1"
                ).fetchone():
                    raise AccountsAlreadyInitialized("La configuración inicial ya se completó")
                connection.execute(
                    """
                    INSERT INTO web_users
                        (user_id, email, name, password_hash, workspace_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, normalized_email, normalized_name, password_hash, workspace, created_at),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                if "email" in str(exc).lower():
                    raise AccountAlreadyExists("Ya existe una cuenta con ese correo") from None
                raise AccountError("No se pudo crear la cuenta") from None
            except Exception:
                connection.rollback()
                raise
        return {
            "user_id": user_id,
            "email": normalized_email,
            "name": normalized_name,
            "workspace_id": workspace,
            "created_at": created_at,
        }

    def authenticate(self, email: str, password: str, client_key: str) -> LoginResult:
        try:
            normalized_email = _normalize_email(email)
        except AccountError:
            normalized_email = "invalid"
        now = time.time()
        identifier = hashlib.sha256(
            (normalized_email + "\0" + client_key[:256]).encode("utf-8")
        ).hexdigest()
        with closing(self._connect()) as connection:
            attempt = connection.execute(
                "SELECT blocked_until FROM web_login_attempts WHERE identifier_hash = ?",
                (identifier,),
            ).fetchone()
            if attempt and float(attempt[0]) > now:
                return LoginResult(None, rate_limited=True)
            row = connection.execute(
                """
                SELECT user_id, email, name, password_hash, workspace_id, created_at, is_active
                FROM web_users WHERE email = ? COLLATE NOCASE
                """,
                (normalized_email,),
            ).fetchone()

        valid_password = _verify_password(password, row["password_hash"] if row else None)
        if not row or not bool(row["is_active"]) or not valid_password:
            locked = self._record_failure(identifier, now)
            return LoginResult(None, rate_limited=locked)
        with closing(self._connect()) as connection:
            connection.execute(
                "DELETE FROM web_login_attempts WHERE identifier_hash = ?", (identifier,)
            )
        return LoginResult(self._public_user(row))

    def _record_failure(self, identifier: str, now: float) -> bool:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT failures, window_started FROM web_login_attempts
                WHERE identifier_hash = ?
                """,
                (identifier,),
            ).fetchone()
            if not row or now - float(row["window_started"]) >= _LOGIN_WINDOW_SECONDS:
                failures, window_started = 1, now
            else:
                failures, window_started = int(row["failures"]) + 1, float(row["window_started"])
            blocked_until = now + _LOGIN_LOCK_SECONDS if failures >= _LOGIN_FAILURE_LIMIT else 0
            connection.execute(
                """
                INSERT INTO web_login_attempts
                    (identifier_hash, failures, window_started, blocked_until)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(identifier_hash) DO UPDATE SET
                    failures = excluded.failures,
                    window_started = excluded.window_started,
                    blocked_until = excluded.blocked_until
                """,
                (identifier, failures, window_started, blocked_until),
            )
            connection.commit()
        return blocked_until > now

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @staticmethod
    def _public_user(row: sqlite3.Row) -> dict[str, str]:
        return {
            "user_id": str(row["user_id"]),
            "email": str(row["email"]),
            "name": str(row["name"]),
            "workspace_id": str(row["workspace_id"]),
            "created_at": str(row["created_at"]),
        }


def _normalize_email(email: str) -> str:
    if not isinstance(email, str):
        raise AccountError("Escribe un correo válido")
    normalized = email.strip().casefold()
    if (
        len(normalized) > _EMAIL_LIMIT
        or normalized.count("@") != 1
        or any(character.isspace() or ord(character) < 32 for character in normalized)
    ):
        raise AccountError("Escribe un correo válido")
    local, domain = normalized.split("@", 1)
    if not local or len(local) > 64 or not domain or "." not in domain:
        raise AccountError("Escribe un correo válido")
    return normalized


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise AccountError("Escribe tu nombre")
    normalized = " ".join(name.split())
    if not normalized or len(normalized) > 120:
        raise AccountError("El nombre debe tener entre 1 y 120 caracteres")
    return normalized


def _validate_password(password: str) -> bytes:
    if not isinstance(password, str):
        raise AccountError("La contraseña no es válida")
    encoded = password.encode("utf-8")
    if not encoded or len(encoded) > _PASSWORD_LIMIT:
        raise AccountError("La contraseña debe tener entre 1 y 1024 bytes")
    return encoded


def _hash_password(password: str) -> str:
    encoded = _validate_password(password)
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        encoded,
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_BYTES,
        maxmem=64 * 1024 * 1024,
    )
    return (
        "scrypt$1$"
        + base64.urlsafe_b64encode(salt).decode("ascii")
        + "$"
        + base64.urlsafe_b64encode(digest).decode("ascii")
    )


def _verify_password(password: str, stored_hash: str | None) -> bool:
    try:
        encoded = _validate_password(password)
    except AccountError:
        encoded = b""
    salt = _DUMMY_SALT
    expected = _DUMMY_HASH
    if stored_hash:
        parts = stored_hash.split("$")
        if len(parts) == 4 and parts[0] == "scrypt" and parts[1] == "1":
            try:
                salt = base64.urlsafe_b64decode(parts[2].encode("ascii"))
                expected = base64.urlsafe_b64decode(parts[3].encode("ascii"))
            except (ValueError, UnicodeEncodeError):
                salt, expected = _DUMMY_SALT, _DUMMY_HASH
    try:
        actual = hashlib.scrypt(
            encoded,
            salt=salt,
            n=_SCRYPT_N,
            r=_SCRYPT_R,
            p=_SCRYPT_P,
            dklen=_SCRYPT_BYTES,
            maxmem=64 * 1024 * 1024,
        )
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(actual, expected)


def _restrict_database_permissions(database_path: Path) -> None:
    """Restrict the account database through an opened file descriptor."""
    if not hasattr(os, "fchmod"):
        return
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if directory_flag and os.open in os.supports_dir_fd:
        parent_fd = os.open(
            database_path.parent,
            os.O_RDONLY | directory_flag | no_follow,
        )
        try:
            database_fd = os.open(
                database_path.name,
                os.O_RDONLY | no_follow,
                dir_fd=parent_fd,
            )
        finally:
            os.close(parent_fd)
    else:
        database_fd = os.open(database_path, os.O_RDONLY | no_follow)
    try:
        os.fchmod(database_fd, 0o600)
    finally:
        os.close(database_fd)


def _restrict_directory_permissions(directory_path: Path) -> None:
    """Keep SQLite sidecar files inaccessible to other local accounts."""
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if not directory_flag or not hasattr(os, "fchmod"):
        return
    directory_fd = os.open(directory_path, os.O_RDONLY | directory_flag | no_follow)
    try:
        os.fchmod(directory_fd, 0o700)
    finally:
        os.close(directory_fd)


def main() -> None:
    """Create local web accounts from the deployment host or container."""
    parser = argparse.ArgumentParser(description="Manage DiscourseLens web accounts")
    parser.add_argument(
        "--project-dir",
        default=os.environ.get("DISCOURSELENS_PROJECT_DIR"),
        help="Persistent project storage directory",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("count", help="Print the number of local accounts")
    create = subparsers.add_parser("create-user", help="Create a local account")
    create.add_argument("--name", required=True)
    create.add_argument("--email", required=True)
    create.add_argument("--workspace-id", help=argparse.SUPPRESS)
    create.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read the password from standard input instead of prompting",
    )
    args = parser.parse_args()
    if not args.project_dir:
        parser.error("set --project-dir or DISCOURSELENS_PROJECT_DIR")
    store = WebAccountStore(args.project_dir)
    store.initialize()
    if args.command == "count":
        print(store.count())
        return
    if args.password_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = getpass.getpass("Contraseña: ")
        confirmation = getpass.getpass("Repite la contraseña: ")
        if password != confirmation:
            parser.error("las contraseñas no coinciden")
    user = store.create_user(
        email=args.email,
        name=args.name,
        password=password,
        workspace_id=args.workspace_id,
    )
    print(f"Cuenta creada: {user['name']} <{user['email']}>")


if __name__ == "__main__":
    main()
