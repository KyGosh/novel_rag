"""按不可信数据读取 zip 容器（epub）：不解压到磁盘，限制文件数与解压后大小，拒绝异常路径和加密内容。"""

from __future__ import annotations

import posixpath
import zipfile
from pathlib import Path

from lxml import etree

from novel_rag.ingestion.ir import IngestError

# 字体混淆不是 DRM：EPUB 规范允许对嵌入字体做混淆，其余资源被加密才视为 DRM
_FONT_OBFUSCATION = {"http://www.idpf.org/2008/embedding", "http://ns.adobe.com/pdf/enc#RC"}

XML_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)


class SafeZip:
    def __init__(self, path: Path, *, max_files: int, max_total_bytes: int, max_member_bytes: int):
        try:
            self._zf = zipfile.ZipFile(path)
        except zipfile.BadZipFile as e:
            raise IngestError(f"不是有效的 zip 容器：{e}") from e
        infos = self._zf.infolist()
        if len(infos) > max_files:
            raise IngestError(f"容器内文件数 {len(infos)} 超过上限 {max_files}")
        total = 0
        self._infos: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            name = info.filename
            norm = posixpath.normpath(name)
            if name.startswith(("/", "\\")) or norm.startswith("..") or ":" in name.split("/")[0]:
                raise IngestError(f"容器内存在非法路径：{name!r}")
            if info.flag_bits & 0x1:
                raise IngestError(f"容器内文件被加密（疑似 DRM）：{name}")
            if info.file_size > max_member_bytes:
                raise IngestError(f"{name} 解压后 {info.file_size} 字节，超过单文件上限 {max_member_bytes}")
            total += info.file_size
            if not info.is_dir():
                self._infos[norm] = info
        if total > max_total_bytes:
            raise IngestError(f"解压后总大小 {total} 字节，超过上限 {max_total_bytes}")
        self.file_count = len(self._infos)
        self.total_bytes = total
        self._max_member_bytes = max_member_bytes
        self._check_drm()

    def names(self) -> list[str]:
        return list(self._infos)

    def exists(self, name: str) -> bool:
        return posixpath.normpath(name) in self._infos

    def read(self, name: str) -> bytes:
        info = self._infos.get(posixpath.normpath(name))
        if info is None:
            raise KeyError(name)
        # 不信任头部声明的大小：按上限截断读取，实际超出则报错
        with self._zf.open(info) as f:
            data = f.read(self._max_member_bytes + 1)
        if len(data) > self._max_member_bytes:
            raise IngestError(f"{name} 实际解压大小超过单文件上限")
        return data

    def read_xml(self, name: str) -> etree._Element:
        try:
            return etree.fromstring(self.read(name), XML_PARSER)
        except etree.XMLSyntaxError as e:
            raise IngestError(f"{name} 不是合法的 XML：{e}") from e

    def _check_drm(self) -> None:
        if self.exists("META-INF/rights.xml"):
            raise IngestError("检测到 META-INF/rights.xml，文件受 DRM 保护，无法解析")
        if not self.exists("META-INF/encryption.xml"):
            return
        root = self.read_xml("META-INF/encryption.xml")
        for method in root.iter("{*}EncryptionMethod"):
            if method.get("Algorithm") not in _FONT_OBFUSCATION:
                raise IngestError("META-INF/encryption.xml 中存在非字体资源的加密，文件受 DRM 保护，无法解析")

    def close(self) -> None:
        self._zf.close()

    def __enter__(self) -> SafeZip:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
