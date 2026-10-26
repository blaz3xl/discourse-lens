# Browser workspace architecture

The browser workspace is one FastAPI application (`src/discourselens/web.py`)
that serves a single HTML document and a JSON API over the shared Core. The
page, its CSS and its JavaScript are one string, `_INDEX_HTML`, with no build
step and no external requests. `tests/test_web.py` asserts substrings of that
string, so keep quoted copy and identifiers stable when refactoring. For hosting,
updating and remote access, see the [web deployment guide](../WEB_DEPLOYMENT.md).

## Information model

- A **project** owns **sources** (documents) and **conversations**. A
  conversation belongs to exactly one project.
- The sidebar is the tree *projects → (sources, conversations)*. The header is a
  breadcrumb, *Project › Conversation*. Opening a project shows its home: its
  sources and a composer that starts a new conversation inside it.
- A conversation row shows its title and, on a second line, its creation moment
  as `(dd/mm/yyyy hh:mm)`. `created_at` is stored in UTC with an offset
  (`datetime.now(UTC).isoformat()`); `createdStamp` renders it in the browser's
  time zone and draws nothing for a value that is not a date. The list is
  ordered by `updated_at`, so creation dates are not monotonic down the list.

## First load

1. If `localStorage` holds `dl:session=1`, a script in `<head>` marks the
   document (`data-session`, so the shell paints instead of the sign-in form) and
   starts up to eight requests while the page is still downloading: projects,
   conversations, model presets, spending, running analyses and, from `dl:last`,
   the sources of the last project and the messages and activity of the last
   conversation. The answers are kept in `window.__dlPrefetch` and each is used
   once, only while it is fresh (20 s).
2. `initAuth` reads `/auth/config`. A signed-in session continues with
   `loadWorkspace({shell: true})`; otherwise the sign-in form is shown.
3. `bootWorkspace` wraps `runBoot` in the loading gate. `runBoot` waits for the
   parallel burst (projects, conversations, running analyses), paints the sidebar,
   then waits for the last project's sources and the last conversation's
   messages. A failure shows `showBootError` with **Reintentar**.
4. **Loading gate** (`bootGate`). From the first step the app container
   (`#app`) is `inert`. After 0.7 s a modal dialog (`#bootDialog`) names the
   pending step of four (session; projects and conversations; sources and last
   conversation; models), stays at least 0.4 s once shown, and closes by itself
   when the last step arrives. The *models* step means the list is in place, not
   that the thinking options, vector indexes and balance have loaded after it.
   - Esc is stopped at the document while the dialog is open, and a close that
     slips through is reopened while the load runs.
   - After 20 s the dialog offers to reload the page.
   - `connected(false)` (sign-out, expired session, stale hint) and a failed
     boot release the gate, so the page is never left inert.
   - The dialog sits outside `#app`: inside an inert ancestor it would be inert
     too and its button could not be used.

## Navigation and state

- `docsCache` and `messagesCache` answer project and conversation switches from
  memory and revalidate in the background. `navSeq` drops answers that arrive
  after the user moved on, and `navigateWorkspace` serialises moves.
- Sidebar rendering reuses DOM nodes (`projectNodes`, plus one row map per
  project), so expanding, collapsing and new rows can animate and rows are not
  rebuilt on every change.
- `localStorage` keys: `dl:session` (a session exists), `dl:last` (`{p, c}`: the
  last project and conversation), `dl:balance` (the last balance that could be
  computed, kept while a newer figure is incomplete), `dl:notices` (the notice
  history below). All are removed on sign-out. The token-mode credential lives
  in `sessionStorage`.

## Toasts and notices

The composer has no footer line. What the workspace says about itself is a
toast: `toast(message, {kind, key, routine})` shows it under the header and the
tools row, so it never covers the project and conversation names. Info toasts
last 4.2 s and errors 7 s (errors are `role="alert"`); a toast with a `key`
replaces the previous one of its channel instead of stacking. `status('chatStatus'
| 'uploadStatus', ...)` remains as an alias that raises a toast; the other
`status(id, ...)` lines (settings, analysis tools) still write into their own panel.

Every toast except a *routine* one (the turn count shown when a conversation
opens) is also recorded by `recordNotice`: newest first, at most 50, and an
identical message within 30 s becomes one entry with `×N`. The bell next to
**Fuentes** (`#openNotices`, label "Avisos") opens `#noticesPanel`, a popover
anchored under it that lists the entries with their time; its badge counts what
arrived since the panel was last open, and routine toasts never count. The
history lives in `localStorage` (`dl:notices`), is validated and capped when it is
loaded, is rendered as text, and is removed at sign-out and when the session is lost.

## Running analyses and reconnecting

A chat turn runs on the server and its progress lives in the process-wide
`_CHAT_ACTIVITY` dictionary: it is lost when the process restarts, entries are
dropped after 3,600 s without updates, there are at most 256 of them, and each
belongs to one project root. `GET /api/chat-activity` lists the running turns;
`GET /api/conversations/{id}/activity` also returns the answer streamed so far
(`partial_text`, at most 120,000 characters). After a reload the sidebar marks
running conversations, a notice offers to reconnect, and `startActivityPolling`
polls every 1.2 s while that conversation is open. A turn is stored in the
conversation only when its answer is complete.

## Transport

`GZipMiddleware(minimum_size=1024)` compresses the page and large JSON; it skips
`text/event-stream`, so streaming stays unbuffered. The page carries an `ETag`
and `Cache-Control: no-cache`, and `If-None-Match` is answered with `304`.
uvicorn speaks HTTP/1.1; a reverse proxy or Funnel may add HTTP/2. A browser
opens at most six HTTP/1.1 connections per origin, so on a direct connection the
requests beyond six queue: keep the first-load burst small.

## Rules for UI code

- Build DOM with `createElement` and `textContent`. The page has no `innerHTML`,
  `outerHTML`, `insertAdjacentHTML`, `document.write`, `eval` or `new Function`.
- Links built from data are validated as `http(s)` and carry
  `rel="noopener noreferrer"` when they open a new tab.
- Requests that change state are checked on the server against the public origin
  (or the request's own host when none is set) by `_ensure_same_origin`.

## Verifying a UI change

The server answers in milliseconds, so latency problems only show on a simulated
link. Run an isolated instance (its own `DISCOURSELENS_PROJECT_DIR`, password
mode, a throwaway account) with a fake OpenAI-compatible streaming model, drive
it with a browser automation tool, delay `**/api/**` and `/auth/**` requests, and
record events with a `MutationObserver` instead of sampling. Worth checking each
time:

- A fast load never shows the dialog; a slow one shows it, blocks clicks and keys,
  and releases only when the history is complete.
- A failed load, a stale `dl:session`, sign-out and a request that never answers
  never leave the page inert.
- Dates are right across a daylight-saving change, a day and a year roll-over,
  and a stored value that is not a date.
- At phone width nothing overflows horizontally and no row's text spills out
  (`scrollHeight > clientHeight`).
- An accessibility scan reports no violations in each state.
