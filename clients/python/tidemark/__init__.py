"""Python client for the Tidemark memory service (standard library only).

    from tidemark import Tidemark
    tm = Tidemark("http://127.0.0.1:18200", api_key="...")
    tm.add("alice", "s1", [{"role": "user", "content": "I adopted a cat named Pixel yesterday."}])
    for hit in tm.search("alice", "What is my cat called?", top_k=5):
        print(hit["score"], hit["content"])
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

__version__ = "1.2.0"
__all__ = ["Tidemark", "TidemarkError"]


class TidemarkError(Exception):
    """An error answer from the service (status code and its message)."""

    def __init__(self, status, message):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


class Tidemark:
    """One Tidemark service. Every method names the user it acts for; a user only
    ever sees and changes their own memories."""

    def __init__(self, url="http://127.0.0.1:18200", api_key=None, timeout=120, retries=5):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries

    # -- transport --------------------------------------------------------
    def _call(self, method, path, body=None, query=None):
        if query:
            path += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        headers = {"Content-Type": "application/json", "User-Agent": f"tidemark-python/{__version__}"}
        if self.api_key:
            headers["Authorization"] = f"Token {self.api_key}"
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read() or b"null")
            except urllib.error.HTTPError as e:
                raw = e.read()
                try:
                    message = json.loads(raw).get("error", raw.decode(errors="replace"))
                except ValueError:
                    message = raw.decode(errors="replace")
                # 429 (busy) and 503 (embedding provider unavailable) are worth retrying.
                if e.code in (429, 503) and attempt < self.retries:
                    time.sleep(float(e.headers.get("Retry-After") or 2 * (attempt + 1)))
                    continue
                raise TidemarkError(e.code, message) from None

    @staticmethod
    def _id(memory_id):
        return urllib.parse.quote(memory_id, safe="")

    # -- Add and Search ---------------------------------------------------
    def add(self, user_id, session_id, messages, request_id=None):
        """Stores messages ({"role", "content", optional "timestamp" in ms, optional "utc_offset_minutes"}). Returns
        after they are durable and searchable. Reusing a request_id is safe: the
        retry is acknowledged without storing twice."""
        return self._call("POST", "/add", {"request_id": request_id or uuid.uuid4().hex, "user_id": user_id,
                                           "session_id": session_id, "messages": messages})

    def search(self, user_id, query, top_k=10, options=None, **settings):
        """The user's most relevant memories for `query`, best first. Extra keyword
        arguments are per-request settings (e.g. max_results=30, passages=True). With passages=True long records
        also carry "passage" (a reading aid; the record's "content" stays authoritative; "start"/"end" are
        code-point offsets, so content[start:end] == passage["text"] in Python)."""
        body = {"query": query, "user_id": user_id, "top_k": top_k, **settings}
        if options:
            body["options"] = options
        return self._call("POST", "/search", body)["data"]

    def search_with_rules(self, user_id, query, top_k=10, options=None, rule_scope=None, **settings):
        """Like search, but returns {"data": [...], "rules": [...]}: the results plus the user's active explicit rules,
        which never take a result slot. rule_scope limits the rules to that scope plus unscoped ones."""
        body = {"query": query, "user_id": user_id, "top_k": top_k, **settings}
        if options:
            body["options"] = options
        if rule_scope is not None:
            body["rule_scope"] = rule_scope
        result = self._call("POST", "/search", body)
        return {"data": result["data"], "rules": result.get("rules", [])}

    # -- Managing memories ------------------------------------------------
    def list(self, user_id, session_id=None, limit=100, offset=0):
        """The user's memories in storage order: {"data", "total", "limit", "offset"}."""
        return self._call("GET", "/memories", query={"user_id": user_id, "session_id": session_id, "limit": limit, "offset": offset})

    def get(self, user_id, memory_id):
        return self._call("GET", f"/memories/{self._id(memory_id)}", query={"user_id": user_id})["data"]

    def update(self, user_id, memory_id, content):
        """Corrects a memory's text (re-embedded; the previous text stays in its history)."""
        return self._call("PATCH", f"/memories/{self._id(memory_id)}", {"user_id": user_id, "content": content})["data"]

    def delete(self, user_id, memory_id):
        return self._call("DELETE", f"/memories/{self._id(memory_id)}", query={"user_id": user_id})

    def history(self, user_id, memory_id):
        return self._call("GET", f"/memories/{self._id(memory_id)}/history", query={"user_id": user_id})["data"]

    # -- Preferences ------------------------------------------------------

    def preferences(self, user_id, all=False):
        """The user's current preferences (with all=True also earlier versions,
        retracted and deleted ones)."""
        return self._call("GET", "/preferences", query={"user_id": user_id, **({"status": "all"} if all else {})})["data"]

    def add_preference(self, user_id, statement, when_to_apply=None, category=None):
        body = {"user_id": user_id, "statement": statement, "when_to_apply": when_to_apply, "category": category}
        return self._call("POST", "/preferences", {k: v for k, v in body.items() if v is not None})["data"]

    def update_preference(self, user_id, preference_id, statement=None, when_to_apply=None, category=None):
        """A new version of the preference; the old one is kept as superseded."""
        body = {"user_id": user_id, "statement": statement, "when_to_apply": when_to_apply, "category": category}
        return self._call("PATCH", f"/preferences/{self._id(preference_id)}", {k: v for k, v in body.items() if v is not None})["data"]

    def delete_preference(self, user_id, preference_id):
        return self._call("DELETE", f"/preferences/{self._id(preference_id)}", query={"user_id": user_id})

    def preference_history(self, user_id, preference_id):
        return self._call("GET", f"/preferences/{self._id(preference_id)}/history", query={"user_id": user_id})["data"]

    # -- Explicit standing rules -----------------------------------------
    def rules(self, user_id, status="active"):
        """The user's rules: status "active" (default), "revoked" or "all"."""
        return self._call("GET", "/rules", query={"user_id": user_id, "status": status})["data"]

    def add_rule(self, user_id, text, scope=None):
        """Adds an active rule ("Don't use Git"), optionally for a scope. An identical active rule is returned as is.
        Each rule's "possible_conflicts" names other active rules that may contradict it: a hint that can be
        incomplete (an empty list does not mean no conflict)."""
        body = {"user_id": user_id, "text": text, **({"scope": scope} if scope is not None else {})}
        return self._call("POST", "/rules", body)["rule"]

    def get_rule(self, user_id, rule_id):
        return self._call("GET", f"/rules/{self._id(rule_id)}", query={"user_id": user_id})

    def update_rule(self, user_id, rule_id, text=None, scope=None):
        """A new version of an active rule; scope="" clears the scope."""
        body = {"user_id": user_id, **({"text": text} if text is not None else {}), **({"scope": scope} if scope is not None else {})}
        return self._call("PATCH", f"/rules/{self._id(rule_id)}", body)["rule"]

    def revoke_rule(self, user_id, rule_id):
        """Revokes a rule; it stays in the history."""
        return self._call("DELETE", f"/rules/{self._id(rule_id)}", query={"user_id": user_id})["rule"]

    def purge_rule(self, user_id, rule_id):
        """Erases a rule and its history."""
        return self._call("DELETE", f"/rules/{self._id(rule_id)}", query={"user_id": user_id, "purge": "true"})

    def rule_history(self, user_id, rule_id):
        return self._call("GET", f"/rules/{self._id(rule_id)}/history", query={"user_id": user_id})["versions"]

    def delete_user(self, user_id):
        """Deletes everything stored for the user: memories, request log, history, preferences and rules."""
        return self._call("DELETE", f"/users/{self._id(user_id)}")

    def health(self):
        return self._call("GET", "/health")
