"""
Visualize corner and edge extraction from card images.
This helps verify our extraction logic is working correctly.
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(str(Path(__file__).parent))


def extract_corners_visual(img: Image.Image, corner_size: int = 100):
    """Extract corner regions and return them with coordinates."""
    w, h = img.size
    corners = []

    # Order: TL, TR, BL, BR
    positions = [
        ("Top-Left", 0, 0, corner_size, corner_size),
        ("Top-Right", w-corner_size, 0, w, corner_size),
        ("Bottom-Left", 0, h-corner_size, corner_size, h),
        ("Bottom-Right", w-corner_size, h-corner_size, w, h)
    ]

    for label, x1, y1, x2, y2 in positions:
        crop = img.crop((x1, y1, x2, y2))
        corners.append((label, crop, (x1, y1, x2, y2)))

    return corners


def extract_edges_visual(img: Image.Image, edge_width: int = 50):
    """Extract edge regions and return them with coordinates."""
    w, h = img.size
    edges = []

    # Order: Top, Right, Bottom, Left
    positions = [
        ("Top", 0, 0, w, edge_width),
        ("Right", w-edge_width, 0, w, h),
        ("Bottom", 0, h-edge_width, w, h),
        ("Left", 0, 0, edge_width, h)
    ]

    for label, x1, y1, x2, y2 in positions:
        crop = img.crop((x1, y1, x2, y2))
        edges.append((label, crop, (x1, y1, x2, y2)))

    return edges


def visualize_card_extractions(front_path: str, back_path: str = None):
    """
    Visualize corner and edge extractions from card images.

    Args:
        front_path: Path to front card image
        back_path: Optional path to back card image
    """
    # Load images
    front_img = Image.open(front_path).convert('RGB')

    if back_path:
        back_img = Image.open(back_path).convert('RGB')
    else:
        back_img = None

    # Extract regions
    front_corners = extract_corners_visual(front_img)
    front_edges = extract_edges_visual(front_img)

    if back_img:
        back_corners = extract_corners_visual(back_img)
        back_edges = extract_edges_visual(back_img)

    # Create visualization
    num_rows = 2 if back_img is None else 4
    fig = plt.figure(figsize=(20, 5 * num_rows))

    # Row 1: Front card with corner boxes
    ax1 = plt.subplot(num_rows, 5, 1)
    ax1.imshow(front_img)
    ax1.set_title("Front Card\n(Corner Regions)", fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Draw corner boxes
    for label, _, (x1, y1, x2, y2) in front_corners:
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=3, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)

    # Show extracted corners
    for i, (label, crop, _) in enumerate(front_corners):
        ax = plt.subplot(num_rows, 5, i + 2)
        ax.imshow(crop)
        ax.set_title(f"Front {label}\n{crop.size[0]}x{crop.size[1]}px", fontsize=10)
        ax.axis('off')

    # Row 2: Front card with edge boxes
    ax2 = plt.subplot(num_rows, 5, 6)
    ax2.imshow(front_img)
    ax2.set_title("Front Card\n(Edge Regions)", fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Draw edge boxes
    colors = ['blue', 'green', 'orange', 'purple']
    for (label, _, (x1, y1, x2, y2)), color in zip(front_edges, colors):
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=3, edgecolor=color, facecolor='none')
        ax2.add_patch(rect)

    # Show extracted edges
    for i, (label, crop, _) in enumerate(front_edges):
        ax = plt.subplot(num_rows, 5, i + 7)
        ax.imshow(crop)
        ax.set_title(f"Front {label}\n{crop.size[0]}x{crop.size[1]}px", fontsize=10)
        ax.axis('off')

    # If back image provided, repeat for back
    if back_img:
        # Row 3: Back corners
        ax3 = plt.subplot(num_rows, 5, 11)
        ax3.imshow(back_img)
        ax3.set_title("Back Card\n(Corner Regions)", fontsize=12, fontweight='bold')
        ax3.axis('off')

        for label, _, (x1, y1, x2, y2) in back_corners:
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=3, edgecolor='red', facecolor='none')
            ax3.add_patch(rect)

        for i, (label, crop, _) in enumerate(back_corners):
            ax = plt.subplot(num_rows, 5, i + 12)
            ax.imshow(crop)
            ax.set_title(f"Back {label}\n{crop.size[0]}x{crop.size[1]}px", fontsize=10)
            ax.axis('off')

        # Row 4: Back edges
        ax4 = plt.subplot(num_rows, 5, 16)
        ax4.imshow(back_img)
        ax4.set_title("Back Card\n(Edge Regions)", fontsize=12, fontweight='bold')
        ax4.axis('off')

        for (label, _, (x1, y1, x2, y2)), color in zip(back_edges, colors):
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=3, edgecolor=color, facecolor='none')
            ax4.add_patch(rect)

        for i, (label, crop, _) in enumerate(back_edges):
            ax = plt.subplot(num_rows, 5, i + 17)
            ax.imshow(crop)
            ax.set_title(f"Back {label}\n{crop.size[0]}x{crop.size[1]}px", fontsize=10)
            ax.axis('off')

    plt.tight_layout()

    # Save visualization
    output_path = "extraction_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    print(f"\nImage dimensions:")
    print(f"  Front: {front_img.size[0]}x{front_img.size[1]}px")
    if back_img:
        print(f"  Back: {back_img.size[0]}x{back_img.size[1]}px")

    plt.show()


def analyze_extraction_quality(img: Image.Image):
    """Analyze if extraction parameters are appropriate for this image."""
    w, h = img.size

    print(f"\n{'='*60}")
    print(f"EXTRACTION ANALYSIS")
    print(f"{'='*60}")
    print(f"Image size: {w}x{h}px")
    print(f"\nCorner extraction (100x100px):")
    print(f"  - Coverage: {(100/w)*100:.1f}% width, {(100/h)*100:.1f}% height")

    if w < 500 or h < 700:
        print(f"  ⚠️  LOW RESOLUTION - corners might be too small")
    elif w > 2000 or h > 3000:
        print(f"  ⚠️  HIGH RESOLUTION - 100px might miss corner details")
    else:
        print(f"  ✓ Resolution looks good for 100px corners")

    print(f"\nEdge extraction (50px width):")
    print(f"  - Top/Bottom: {w}x50px → resized to 224x224 (aspect ratio {w/50:.1f}:1)")
    print(f"  - Left/Right: 50x{h}px → resized to 224x224 (aspect ratio 1:{h/50:.1f})")

    if w/50 > 10 or h/50 > 10:
        print(f"  ⚠️  SEVERE DISTORTION when resizing to square")
        print(f"  → Consider: wider edges (100-150px) or different resize strategy")
    else:
        print(f"  ✓ Aspect ratio acceptable")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize corner/edge extraction')
    parser.add_argument('--front', type=str, required=True,
                       help='Path to front card image')
    parser.add_argument('--back', type=str, default=None,
                       help='Path to back card image')

    args = parser.parse_args()

    # Load and analyze
    front_img = Image.open(args.front).convert('RGB')
    analyze_extraction_quality(front_img)

    # Visualize
    visualize_card_extractions(args.front, args.back)


if __name__ == "__main__":
    main()
