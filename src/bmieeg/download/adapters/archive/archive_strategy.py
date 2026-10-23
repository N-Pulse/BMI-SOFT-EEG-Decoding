# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import requests

from tqdm import tqdm
from pathlib import PurePosixPath
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from requests.auth import HTTPBasicAuth
from urllib.parse import quote, unquote, urlparse

from .archive_specs import ArchiveSpecs
from ...domain import DownloadStrategy, Registry



# ================================================================
# 1. Section: Functions
# ================================================================
@Registry.register("archive")
@dataclass
class ArchiveStrategy(DownloadStrategy):
    def validate(self, spec: ArchiveSpecs) -> None:
        self._auth = HTTPBasicAuth(spec.username, spec.password)

        response = requests.request(
            method="PROPFIND",
            url=spec._dav_root,
            headers=spec._header(),
            data=spec._body,
            auth=self._auth,
            allow_redirects=False,
        )

        if response.status_code != 207:
            raise ValueError(f"Unexpected status code: {response.status_code}")

    def fetch(self, spec: ArchiveSpecs) -> None:
        self.validate(spec)

        # 1. Makes sure we have a folder to download to
        os.makedirs(spec.dest, exist_ok=True)

        # 2. Gets everyting inside the remote folder
        xml_text = _propfind(spec, self._auth, depth="infinity")
        items = _parse_propfind(spec, xml_text)

        # 3. Iterates over every file
        ignore = set(spec.extra.get("ignore", []))
        for item in tqdm(items, desc="Downloading files", unit="file"):
            # 3.1 Get the relative path
            rel_path = _remote_relative_path(spec.dav_user, item["href"])

            # 3.2 Skip the folder itself (propfin returns the folder as the first entry too)
            if rel_path == spec.url:
                continue

            rel_parts = PurePosixPath(rel_path).parts
            if any(part in ignore for part in rel_parts):
                continue

            # 3.3 Makes sub-folders if needed and downloads the files
            if item["is_dir"]:
                remote_folder = _extract_path_of_interest(rel_path, spec._dav_url)
                folder = os.path.join(spec.dest, remote_folder)
                os.makedirs(folder, exist_ok=True)
            else:
                remote_folder = _extract_path_of_interest(rel_path, spec._dav_url)
                local_path = os.path.join(spec.dest, remote_folder)
                remote_url = f"{spec._base_url}{quote(item['href'])}"
                _download_file(self._auth, remote_url, local_path)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _propfind(spec: ArchiveSpecs, auth: HTTPBasicAuth, depth: int | str = 1) -> str:
    r = requests.request(
        "PROPFIND",
        spec._dav_url,
        headers=spec._header(depth),
        data=spec._body,
        auth=auth,
    )
    r.raise_for_status()
    return r.text

def _parse_propfind(spec: ArchiveSpecs, xml_text: str):
    root = ET.fromstring(xml_text)
    items = []

    for response in root.findall("d:response", spec._namespaces):
        href = response.find("d:href", spec._namespaces)
        propstat = response.find("d:propstat", spec._namespaces)
        if href is None or propstat is None:
            continue

        prop = propstat.find("d:prop", spec._namespaces)
        if prop is None:
            continue

        resourcetype = prop.find("d:resourcetype", spec._namespaces)
        is_dir = (
            resourcetype is not None
            and resourcetype.find("d:collection", spec._namespaces) is not None
        )

        items.append(
            {
                "href": unquote(str(href.text)),
                "is_dir": is_dir,
            }
        )

    return items

def _remote_relative_path(dav_user: str, href: str) -> str:
    parsed = urlparse(href)
    path = parsed.path

    prefix = f"/remote.php/dav/files/{dav_user}/"
    if path.startswith(prefix):
        return path[len(prefix) :].strip("/")

    prefix_no_slash = f"/remote.php/dav/files/{dav_user}"
    if path == prefix_no_slash:
        return ""

    raise ValueError(f"Unexpected href: {href}")

def _extract_path_of_interest(rel_path: str, server_path: str) -> str:
    def split_path(path: str) -> list[str]:
        return [part for part in path.strip("/").split("/") if part]

    rel_parts = split_path(rel_path)
    server_parts = split_path(server_path)

    if not server_parts:
        return "/".join(rel_parts)

    # Try to find the deepest matching part of server_path inside rel_path.
    # It prefers longer contiguous matches first.
    for end in range(len(server_parts), 0, -1):
        for length in range(end, 0, -1):
            candidate = server_parts[end - length : end]

            for start in range(len(rel_parts) - length, -1, -1):
                if rel_parts[start : start + length] == candidate:
                    return "/".join(rel_parts[start + length :])

    raise ValueError(
        f"No folder from server_path was found in rel_path:\n"
        f"rel_path: {rel_path}\n"
        f"server_path: {server_path}"
    )

def _download_file(
    auth: HTTPBasicAuth, remote_url: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with requests.get(remote_url, auth=auth, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
