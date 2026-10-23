# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from urllib.parse import quote
from dataclasses import dataclass, field

from ...domain import DownloadSpec


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ArchiveSpecs(DownloadSpec):
    source: ClassVar[str] = "archive"

    url: str
    username: str
    password: str
    dav_user: str
    filename: str | None = None
    extra: dict = field(default_factory=dict)

    _base_url: str = "https://make-archives.epfl.ch"
    _body: str = """<?xml version="1.0"?>
    <d:propfind xmlns:d="DAV:">
      <d:prop>
        <d:displayname />
        <d:resourcetype />
        <d:getcontentlength />
        <d:getlastmodified />
      </d:prop>
    </d:propfind>
    """
    _namespaces: dict = field(default_factory=lambda: {
        "d": "DAV:",
    })

    def __post_init__(self):
        self._dav_root = f"{self._base_url}/remote.php/dav/files/{self.dav_user}"

        remote_path = self.url.strip("/")
        if remote_path:
            self._dav_url = f"{self._dav_root}/{quote(remote_path)}"
        else:
            self._dav_url = self._dav_root

    def _header(self, depth: int | str = 1) -> dict:
        return {
            "Depth": str(depth),
            "Content-Type": "application/xml",
        }
