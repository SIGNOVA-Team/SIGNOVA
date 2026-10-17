from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
VIDEOS_DIRECTORY = PROJECT_DIRECTORY / "data/raw/isharah500/videos"
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "reports/review_sheets"

SAMPLE_IDS = (
    "04_0249",
    "08_0393",
    "09_0394",
    "10_0225",
    "01_0155",
    "04_0274",
    "00_0429",
    "10_0224",
    "01_0395",
    "01_0394",
)
FRAMES_PER_SAMPLE = 12
COLUMNS = 4
THUMBNAIL_SIZE = (200, 200)
LABEL_HEIGHT = 28

def find_sample_directory(sample_id):
    signer_id = sample_id.split("_")[0]
    signer_directory = VIDEOS_DIRECTORY / signer_id

    candidates = (
        signer_directory / sample_id,
        signer_directory / signer_id / sample_id,
    )
    for directory in candidates:
        if directory.is_dir():
            return directory

    raise FileNotFoundError(
        f"Sample directory not found: {sample_id}"
    )

def select_frames(sample_directory):
    frames = sorted(
        path
        for path in sample_directory.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not frames:
        raise ValueError(
            f"No image frames found in {sample_directory}"
        )
    frame_count = min(FRAMES_PER_SAMPLE, len(frames))
    indices = [
        round(index * (len(frames) - 1) / (frame_count - 1))
        for index in range(frame_count)
    ] if frame_count > 1 else [0]

    return [(index, frames[index]) for index in indices]

def create_contact_sheet(sample_id):
    sample_directory = find_sample_directory(sample_id)
    selected_frames = select_frames(sample_directory)
    rows = (
        len(selected_frames) + COLUMNS - 1
    ) // COLUMNS

    cell_width, cell_height = THUMBNAIL_SIZE
    sheet = Image.new(
        "RGB",
        (
            COLUMNS * cell_width,
            rows * (cell_height + LABEL_HEIGHT),
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for position, (frame_index, frame_path) in enumerate(
        selected_frames
    ):
        column = position % COLUMNS
        row = position // COLUMNS

        x = column * cell_width
        y = row * (cell_height + LABEL_HEIGHT)

        with Image.open(frame_path) as image:
            thumbnail = ImageOps.contain(
                image.convert("RGB"),
                THUMBNAIL_SIZE,
            )
            image_x = x + (cell_width - thumbnail.width) // 2
            image_y = y + (cell_height - thumbnail.height) // 2

            sheet.paste(thumbnail, (image_x, image_y))
        draw.text(
            (x + 8, y + cell_height + 5),
            f"Frame {frame_index + 1}",
            fill="black",
        )
    output_path = OUTPUT_DIRECTORY / f"{sample_id}.jpg"
    sheet.save(output_path, quality=90)

    print(f"Created: {output_path.name}")

def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for sample_id in SAMPLE_IDS:
        create_contact_sheet(sample_id)

    print("\nAll review sheets created successfully.")

if __name__ == "__main__":
    main()
