"""
Render radiology report text as document page images for ColPali indexing.
ColPali retrieves document pages as images, so we need to convert text reports
into visually formatted document images.
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.config import RENDERED_REPORTS_DIR
except ImportError:
    RENDERED_REPORTS_DIR = Path("./data/rendered_reports")

# Page layout settings
PAGE_WIDTH = 896
PAGE_HEIGHT = 896
MARGIN = 60
FONT_SIZE = 18
TITLE_FONT_SIZE = 24
LINE_HEIGHT = 26
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
TITLE_COLOR = (0, 51, 102)
HEADER_COLOR = (51, 102, 153)


def get_font(size=FONT_SIZE, bold=False):
    """Get a font, falling back to default if custom fonts unavailable."""
    try:
        if bold:
            return ImageFont.truetype("arialbd.ttf", size)
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        try:
            if bold:
                return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def wrap_text(text, font, max_width):
    """Wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        try:
            bbox = font.getbbox(test_line)
            width = bbox[2] - bbox[0]
        except AttributeError:
            width = font.getsize(test_line)[0]
        if width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def render_report_as_image(report_text, sample_id="", image_path=None):
    """
    Render a radiology report as a document page image.
    
    Args:
        report_text: The report text to render
        sample_id: Sample identifier for the title
        image_path: Optional path to include image reference
    
    Returns:
        PIL Image of the rendered report page
    """
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    title_font = get_font(TITLE_FONT_SIZE, bold=True)
    header_font = get_font(FONT_SIZE + 2, bold=True)
    body_font = get_font(FONT_SIZE)
    small_font = get_font(FONT_SIZE - 4)
    
    y = MARGIN
    max_text_width = PAGE_WIDTH - 2 * MARGIN
    
    # Draw title
    title = f"RADIOLOGY REPORT — {sample_id}" if sample_id else "RADIOLOGY REPORT"
    draw.text((MARGIN, y), title, fill=TITLE_COLOR, font=title_font)
    y += TITLE_FONT_SIZE + 10
    
    # Draw separator line
    draw.line([(MARGIN, y), (PAGE_WIDTH - MARGIN, y)], fill=HEADER_COLOR, width=2)
    y += 15
    
    # Draw report text
    if report_text:
        # Try to detect sections
        sections = split_into_sections(report_text)
        
        for section_title, section_text in sections:
            if y > PAGE_HEIGHT - MARGIN - LINE_HEIGHT:
                break
            
            if section_title:
                draw.text((MARGIN, y), section_title.upper(), fill=HEADER_COLOR, font=header_font)
                y += LINE_HEIGHT + 5
            
            lines = wrap_text(section_text, body_font, max_text_width)
            for line in lines:
                if y > PAGE_HEIGHT - MARGIN - LINE_HEIGHT:
                    draw.text((MARGIN, y), "... [continued]", fill=(128, 128, 128), font=small_font)
                    break
                draw.text((MARGIN, y), line, fill=TEXT_COLOR, font=body_font)
                y += LINE_HEIGHT
            y += 10
    
    # Draw footer
    footer_y = PAGE_HEIGHT - MARGIN
    draw.line([(MARGIN, footer_y - 5), (PAGE_WIDTH - MARGIN, footer_y - 5)], fill=(200, 200, 200), width=1)
    draw.text((MARGIN, footer_y), "MIMIC-CXR Dataset | Academic Use Only", fill=(150, 150, 150), font=small_font)
    
    return img


def split_into_sections(text):
    """
    Try to split report text into sections (Findings, Impression, etc.).
    Returns list of (title, content) tuples.
    """
    import re
    
    section_headers = [
        r'(?i)(findings?)[:\s]',
        r'(?i)(impression)[:\s]',
        r'(?i)(conclusion)[:\s]',
        r'(?i)(indication)[:\s]',
        r'(?i)(technique)[:\s]',
        r'(?i)(comparison)[:\s]',
        r'(?i)(history)[:\s]',
    ]
    
    # Check if text has section headers
    has_sections = any(re.search(p, text) for p in section_headers)
    
    if not has_sections:
        return [("Report", text)]
    
    sections = []
    remaining = text
    
    for pattern in section_headers:
        match = re.search(pattern, remaining)
        if match:
            before = remaining[:match.start()].strip()
            if before and not sections:
                sections.append(("", before))
            section_name = match.group(1)
            # Find the end of this section (next section header or end)
            next_match = None
            for p2 in section_headers:
                m2 = re.search(p2, remaining[match.end():])
                if m2:
                    if next_match is None or m2.start() < next_match.start():
                        next_match = m2
            if next_match:
                content = remaining[match.end():match.end() + next_match.start()].strip()
            else:
                content = remaining[match.end():].strip()
            if content:
                sections.append((section_name, content))
    
    if not sections:
        return [("Report", text)]
    
    return sections


def render_all_reports(df, output_dir=None):
    """
    Render all reports in a DataFrame as document page images.
    
    Args:
        df: DataFrame with 'report_text' and 'sample_id' columns
        output_dir: Directory to save rendered images
    
    Returns:
        List of output image paths
    """
    if output_dir is None:
        output_dir = RENDERED_REPORTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rendering reports"):
        sample_id = row.get("sample_id", f"cxr_{idx:05d}")
        report_text = row.get("report_text", "")
        image_path = row.get("image_path", None)
        
        img = render_report_as_image(report_text, sample_id, image_path)
        
        out_path = output_dir / f"{sample_id}_report.png"
        img.save(out_path, "PNG")
        paths.append(str(out_path))
    
    logger.info(f"Rendered {len(paths)} report pages to {output_dir}")
    return paths


if __name__ == "__main__":
    test_report = (
        "FINDINGS: The heart is mildly enlarged. Mediastinal contours are normal. "
        "There is a small left pleural effusion. No pneumothorax. The lungs show "
        "bibasilar atelectasis. No focal consolidation. No acute osseous abnormality. "
        "IMPRESSION: Mild cardiomegaly with small left pleural effusion and bibasilar atelectasis."
    )
    img = render_report_as_image(test_report, "TEST_001")
    img.save("test_rendered_report.png")
    print("Saved test_rendered_report.png")
