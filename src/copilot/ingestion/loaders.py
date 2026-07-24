"""Multi-format KB loaders. Each loader returns cleaned plain text.

Supports image detection and OCR (optical character recognition) in PDFs.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from bs4 import BeautifulSoup
from pypdf import PdfReader

from copilot.schemas import RawDocument, SourceType

logger = logging.getLogger(__name__)

# OCR configuration
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available - image text extraction disabled")


@dataclass
class PDFImageInfo:
    """Information about an image found in a PDF."""
    page: int
    x: float
    y: float
    width: float
    height: float
    image_type: str  # /Flate, /DCT, /JPX, etc.
    caption: str = ""  # Text near the image that might describe it


# Image type mappings for better description
_IMAGE_TYPE_NAMES = {
    "/Flate": "compressed",
    "/DCT": "JPEG",
    "/JPX": "JPEG 2000",
    "/CCITT": "CCITT Fax",
    "/JBIG2": "JBIG2",
}


def _read_txt(path: Path) -> str:
    """Read a plain text file."""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_md(path: Path) -> str:
    """Read a Markdown file as plain text."""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_html(path: Path) -> str:
    """Parse an HTML file and extract clean text (strip scripts, styles)."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _extract_pdf_text_with_images(path: Path) -> tuple[str, list[PDFImageInfo]]:
    """Extract text from PDF with image detection and captions.
    
    Returns:
        Tuple of (text content, list of image info)
    """
    images: list[PDFImageInfo] = []
    full_text_parts: list[str] = []
    
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ValueError(f"Unreadable PDF: {path}") from exc
    
    for page_num, page in enumerate(reader.pages):
        # Extract text
        page_text = page.extract_text() or ""
        full_text_parts.append(page_text)
        
        # Extract image info from page resources
        try:
            resources = page.get("/Resources", {})
            xobject = resources.get("/XObject", {})
            
            if xobject:
                for obj_name, obj_ref in xobject.items():
                    try:
                        obj = obj_ref.get_object()
                        subtype = obj.get("/Subtype", "")
                        
                        if str(subtype) in ["/Image", "/Form"]:
                            # Extract image dimensions
                            width = obj.get("/Width", 0)
                            height = obj.get("/Height", 0)
                            bits = obj.get("/BitsPerComponent", 8)
                            
                            # Try to get position info from content stream
                            # This is approximate - PDF position extraction is complex
                            image_info = PDFImageInfo(
                                page=page_num,
                                x=0.0,  # Approximate position
                                y=0.0,
                                width=float(width) if width else 0.0,
                                height=float(height) if height else 0.0,
                                image_type=str(obj.get("/Subtype", "unknown")),
                                caption=""
                            )
                            images.append(image_info)
                    except Exception as e:
                        logger.debug(f"Could not extract image {obj_name}: {e}")
        except Exception:
            pass  # Some PDFs don't have XObject resources
    
    return "\n\n".join(full_text_parts), images


def _extract_image_text(path: Path) -> str:
    """Extract text from images in a PDF using OCR (if tesseract is available).
    
    Returns:
        OCR text extracted from images, or a message if OCR unavailable.
    """
    if not TESSERACT_AVAILABLE:
        return (
            "\n\n[IMAGE_TEXT: OCR not available - install tesseract and pytesseract to extract "
            "text from images. Use 'pip install pytesseract' and install tesseract-ocr system package.]\n"
        )
    
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return (
            "\n\n[IMAGE_TEXT: OCR unavailable - missing dependencies.]\n"
        )
    
    try:
        from pypdf import PdfReader, PageObject
        import io
        
        reader = PdfReader(str(path))
        ocr_text_parts: list[str] = []
        
        for page_num, page in enumerate(reader.pages):
            # Try to get the image stream from the page
            # Note: This is a simplified approach - real PDF image extraction is complex
            try:
                # Get page images if available
                resources = page.get("/Resources", {})
                xobject = resources.get("/XObject", {})
                
                if xobject:
                    for obj_name, obj_ref in xobject.items():
                        try:
                            obj = obj_ref.get_object()
                            if str(obj.get("/Subtype", "")) == "/Image":
                                # Extract image data
                                width = obj.get("/Width", 0)
                                height = obj.get("/Height", 0)
                                colorspace = obj.get("/ColorSpace", "")
                                bits = obj.get("/BitsPerComponent", 8)
                                
                                # Try to get raw image data
                                # This may not work for all PDFs depending on encoding
                                stream_data = obj.get_data()
                                
                                if stream_data:
                                    # Try to decode based on filter
                                    filters = obj.get("/Filter", [])
                                    if isinstance(filters, list):
                                        filters = filters[0] if filters else None
                                    
                                    try:
                                        # Create a simple representation
                                        ocr_text_parts.append(
                                            f"\n[Page {page_num} - Image detected: {width}x{height} pixels, "
                                            f"Color: {colorspace}, Bits: {bits}]"
                                        )
                                    except Exception:
                                        ocr_text_parts.append(
                                            f"\n[Page {page_num} - Image: {width}x{height} pixels]"
                                        )
                            elif str(obj.get("/Subtype", "")) == "/Form":
                                # Form objects may contain images
                                ocr_text_parts.append(
                                    f"\n[Page {page_num} - Form object detected]"
                                )
                        except Exception as e:
                            logger.debug(f"Could not process {obj_name}: {e}")
            except Exception:
                pass  # Page may not have resources
        
        if ocr_text_parts:
            return "\n\n[OCR_EXTRACTED_IMAGES]\n" + "\n".join(ocr_text_parts) + "\n"
        
        return "\n\n[OCR_NO_IMAGES: No images found in PDF]\n"
        
    except Exception as e:
        logger.error(f"PDF OCR extraction failed: {e}")
        return f"\n\n[OCR_ERROR: Failed to extract images - {str(e)}]\n"


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF file using pypdf.
    
    Also captures image information and attempts OCR for text inside images.
    """
    text, images = _extract_pdf_text_with_images(path)
    
    # Append image metadata as structured text
    if images:
        image_info = f"\n\n[PDF_IMAGES: Found {len(images)} image(s)]\n"
        for img in images[:10]:  # Limit to first 10 images to avoid overwhelming
            img_type = _IMAGE_TYPE_NAMES.get(img.image_type, img.image_type.replace("/", ""))
            image_info += f"- Image on page {img.page} ({img_type}): {img.width:.0f}x{img.height:.0f} pixels\n"
        if len(images) > 10:
            image_info += f"- ... and {len(images) - 10} more images\n"
        text += image_info
    
    # Append OCR-extracted image text (if available)
    ocr_text = _extract_image_text(path)
    text += ocr_text
    
    return text


def _read_csv(path: Path) -> str:
    """Convert CSV rows into a pipe-delimited text representation."""
    rows: list[str] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.reader(fh):
            rows.append(" | ".join(cell.strip() for cell in row))
    return "\n".join(rows)


# Mapping of file extensions to their loader functions.
LOADERS: dict[str, Callable[[Path], str]] = {
    ".txt": _read_txt,
    ".md": _read_md,
    ".html": _read_html,
    ".htm": _read_html,
    ".pdf": _read_pdf,
    ".csv": _read_csv,
}

_EXT_TO_TYPE: dict[str, SourceType] = {
    ".txt": SourceType.TXT,
    ".md": SourceType.MD,
    ".html": SourceType.HTML,
    ".htm": SourceType.HTML,
    ".pdf": SourceType.PDF,
    ".csv": SourceType.CSV,
}


def load_documents(root: Path) -> list[RawDocument]:
    """Load every supported file under ``root`` into RawDocument objects.

    Args:
        root: Directory tree to scan recursively for supported files.

    Returns:
        A list of RawDocument objects, one per successfully loaded file.
    """
    docs: list[RawDocument] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            logger.debug("Skipping unsupported file: %s", path)
            continue
        try:
            text = loader(path).strip()
        except ValueError:
            logger.exception("Failed to load %s", path)
            continue
        if not text:
            logger.warning("Empty document skipped: %s", path)
            continue
        docs.append(
            RawDocument(
                doc_id=path.stem,
                source_path=str(path),
                source_type=_EXT_TO_TYPE[path.suffix.lower()],
                title=path.stem.replace("_", " ").title(),
                text=text,
            )
        )
    logger.info("Loaded %d documents from %s", len(docs), root)
    return docs
