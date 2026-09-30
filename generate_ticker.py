import os
import argparse
from PIL import Image, ImageDraw, ImageFont


def get_best_font(pixel_height: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Attempts to load a thick, highly legible font for the matrix."""
    font_candidates = [
        "arialbd.ttf",       # Windows / Standard
        "DejaVuSans-Bold.ttf",# Linux
        "Helvetica-Bold.ttc", # Mac
        "FreeSansBold.ttf",
    ]
    for font_name in font_candidates:
        try:
            # We request the exact pixel height for perfect 1:1 mapping
            return ImageFont.truetype(font_name, pixel_height)
        except OSError:
            continue
    return ImageFont.load_default()


def generate_pro_led_ticker(
    output_path: str,
    text: str,
    direction: str = "left",
    speed: int = 1,           # 1 LED column shift per frame (smoothest)
    led_rows: int = 24,       # Higher row count = much clearer text
    led_size: int = 6,        # Size of the LED circle
    led_gap: int = 2,         # Space between LEDs
    canvas_width: int = 800,  # Target panel width before snapping to LED grid
    # color_on: tuple = (10, 255, 120),  # Neon Matrix Green
    # color_off: tuple = (15, 22, 18),   # Dark unlit LED
    # bg_color: tuple = (5, 8, 5)        # Panel background

    # color_on = (255, 30, 30),     # Vivid Bright Red
    # color_off = (25, 10, 10) ,    # Dim Dark Red
    # bg_color = (8, 3, 3)         # Pitch Black / Dark Red Tint

    # color_on = (255, 170, 0),     # Bright Amber Gold
    # color_off = (30, 20, 5),      # Dim Unlit Amber
    # bg_color = (10, 8, 2)        # Black Background

    color_on = (0, 220, 255),     # Bright Neon Blue
    color_off = (10, 25, 35),     # Dim Unlit Blue
    bg_color = (3, 8, 12)        # Dark Navy Panel
) -> None:
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1. Calculate Grid Pitch and Canvas Size
    pitch = led_size + led_gap
    canvas_height = led_rows * pitch
    
    # Target a width of around 800px, but snap perfectly to the LED grid
    cols = canvas_width // pitch
    canvas_width = cols * pitch

    # 2. Create the Pixel-Perfect Text Mask
    # We render the text such that 1 pixel exactly equals 1 LED.
    # We use size=led_rows so the font fits perfectly top-to-bottom.
    font = get_best_font(int(led_rows * 0.9))
    
    # Get exact text dimensions
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # The mask needs padding equal to the screen columns so it scrolls fully in and out
    mask_width = text_w + (cols * 2)
    
    # '1' mode creates a pure binary (black/white) image. No blurry anti-aliasing.
    text_mask = Image.new("1", (mask_width, led_rows), 0)
    draw_mask = ImageDraw.Draw(text_mask)

    # Center text vertically on the mask
    text_y = (led_rows - text_h) // 2 - bbox[1]
    
    # Draw text onto the mask
    draw_mask.text((cols, text_y), text, fill=1, font=font)

    # 3. Pre-render a single blank LED panel frame to save processing time
    base_frame = Image.new("RGB", (canvas_width, canvas_height), bg_color)
    base_draw = ImageDraw.Draw(base_frame)
    
    for r in range(led_rows):
        for c in range(cols):
            x, y = c * pitch, r * pitch
            # Draw the "off" LEDs once
            base_draw.ellipse([x, y, x + led_size, y + led_size], fill=color_off)

    # 4. Generate Animation Frames
    frames = []
    # Total shifts needed to move the text completely across the screen
    total_scroll_distance = text_w + cols

    print("Rendering frames, please wait...")
    for offset in range(0, total_scroll_distance, speed):
        # Start with a fresh copy of the blank panel
        frame = base_frame.copy()
        draw = ImageDraw.Draw(frame)

        start_col = offset if direction == "left" else (total_scroll_distance - offset)

        # Only draw the "ON" LEDs for efficiency
        for r in range(led_rows):
            for c in range(cols):
                mask_x = start_col + c
                
                # Check if the pixel on the mask is white (1)
                if 0 <= mask_x < mask_width:
                    if text_mask.getpixel((mask_x, r)):
                        x, y = c * pitch, r * pitch
                        
                        # Outer Glow (slightly darker base)
                        glow_color = (int(color_on[0]*0.6), int(color_on[1]*0.6), int(color_on[2]*0.6))
                        draw.ellipse([x, y, x + led_size, y + led_size], fill=glow_color)
                        
                        # Inner Bright Core (simulates a real LED diode)
                        core_offset = max(1, led_size // 4)
                        draw.ellipse(
                            [x + core_offset, y + core_offset, x + led_size - core_offset, y + led_size - core_offset], 
                            fill=color_on
                        )

        # Quantize colors to reduce GIF file size while keeping it visually flawless
        frames.append(frame.quantize(colors=16))

    # 5. Save the GIF
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=30, # Faster frame rate for smooth 1-pixel scrolling
            loop=0,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ultra-premium LED ticker GIF")
    parser.add_argument("--output", "-o", default="assets/demo.gif", help="Output GIF path")
    parser.add_argument("--text", "-t", default="LED Ticker", help="Ticker text")
    parser.add_argument("--direction", "-d", choices=("left", "right"), default="left", help="Scroll direction")
    parser.add_argument("--speed", "-s", type=int, default=2, help="Scroll speed (LED columns per frame)")
    args = parser.parse_args()

    out_path = os.path.normpath(args.output)
    generate_pro_led_ticker(out_path, text=args.text, direction=args.direction, speed=args.speed)
    print(f"Success! High-resolution LED GIF saved at: {out_path}")