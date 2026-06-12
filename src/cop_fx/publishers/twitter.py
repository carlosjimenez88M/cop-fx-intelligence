"""Twitter/X publisher using Tweepy v4 (API v2)."""

from __future__ import annotations

from cop_fx.config.settings import get_settings
from cop_fx.logger import get_logger

logger = get_logger(__name__)


class TwitterPublisher:
    """Posts tweets via the Twitter API v2 using OAuth 1.0a + Bearer Token."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = self._build_client()

    def _build_client(self):  # type: ignore[return]
        import tweepy  # deferred so the class can be imported without tweepy installed

        s = self._settings
        missing = [
            name
            for name, val in [
                ("TWITTER_API_KEY", s.twitter_api_key),
                ("TWITTER_API_SECRET", s.twitter_api_secret),
                ("TWITTER_ACCESS_TOKEN", s.twitter_access_token),
                ("TWITTER_ACCESS_TOKEN_SECRET", s.twitter_access_token_secret),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(f"Missing Twitter credentials: {missing}")

        return tweepy.Client(
            bearer_token=s.twitter_bearer_token.get_secret_value() if s.twitter_bearer_token else None,
            consumer_key=s.twitter_api_key.get_secret_value(),  # type: ignore[union-attr]
            consumer_secret=s.twitter_api_secret.get_secret_value(),  # type: ignore[union-attr]
            access_token=s.twitter_access_token.get_secret_value(),  # type: ignore[union-attr]
            access_token_secret=s.twitter_access_token_secret.get_secret_value(),  # type: ignore[union-attr]
            wait_on_rate_limit=True,
        )

    def post(self, text: str) -> str:
        """Post ``text`` and return the tweet ID."""
        if len(text) > 280:
            raise ValueError(f"Tweet too long ({len(text)} chars): {text[:50]}…")

        if not self._settings.twitter_enabled:
            logger.info("Twitter disabled — would have posted: %s", text[:80])
            return "dry-run"

        resp = self._client.create_tweet(text=text)
        tweet_id = str(resp.data["id"])  # type: ignore[index]
        logger.info("Tweet posted: id=%s", tweet_id)
        return tweet_id

    def delete(self, tweet_id: str) -> bool:
        """Delete a tweet by ID. Returns True if successful."""
        resp = self._client.delete_tweet(tweet_id)
        success = bool(resp.data.get("deleted"))  # type: ignore[union-attr]
        logger.info("Tweet %s deleted=%s", tweet_id, success)
        return success
