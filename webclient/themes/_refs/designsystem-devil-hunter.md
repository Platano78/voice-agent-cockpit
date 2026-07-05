## Brand & Style
The design system is rooted in a "Visceral Functionalism" aesthetic—a collision of urban-punk chaos and industrial precision. It is designed to evoke the raw, high-stakes energy of a hunter operating in a decaying metropolis. The target audience values speed, directness, and an unapologetic, gritty atmosphere.

The style leans heavily into **Brutalism** with a manga-inspired editorial layer. It rejects soft gradients and polite affordances in favor of high-contrast layouts, ink-bleed textures, and mechanical UI elements. The interface should feel like a redacted government file or a street-poster slapped onto a concrete wall—urgent, dangerous, yet meticulously organized for high-stress utility.

## Layout & Spacing
The layout follows a **Rigid Grid** system inspired by manga panels. Use a 12-column grid for desktop and a 4-column grid for mobile. 

Spacing is tight and aggressive. Elements should often feel "packed" into the screen. Vertical rhythm is controlled by a 4px baseline unit. 

**Structural Motifs:**
- **Paneling:** Break content into distinct boxes with heavy 2px or 3px borders. 
- **The "Broken" Grid:** Occasionally allow an image or a headline to "break" the panel border, overlapping the gutter to create a sense of kinetic movement.
- **Hazard Dividers:** Use 45-degree diagonal stripes (Safety Orange and Black) for major section breaks or loading states.

## Elevation & Depth
This design system rejects soft ambient shadows. Depth is communicated through **Hard Offsets** and **Tonal Layering**:

- **Hard Shadows:** Instead of blurs, use a solid block of color (Secondary or Neutral) offset by 4px or 8px behind an element to give it "weight."
- **Taped Layers:** Cards and panels should look like they are taped or stapled onto the background. Use small rectangular "tape" elements at corners.
- **Halftone Blurs:** If a background needs to be pushed back, use a 50% halftone dot pattern overlay rather than a Gaussian blur.
- **Inverted Surfaces:** For high-priority callouts, invert the color scheme (e.g., Orange background with Black text) to make the element "pop" forward in the hierarchy.

## Components
- **Buttons:** Solid blocks of Safety Orange. Text is JetBrains Mono, All-caps, centered. On hover, the button should "shake" slightly (2px jitter) or invert colors.
- **Input Fields:** A heavy 2px bottom border only. Use JetBrains Mono for input text. Labels should look like "Field Tags" (small, boxed-in text).
- **Cards:** Heavy black borders (#0a0a0a) with a subtle "scratchy" inner border. Headers of cards are separated by a thin horizontal line or a row of "X" marks.
- **Chips/Tags:** Small rectangular boxes with a "staple" icon or a vertical hazard stripe on the left edge.
- **Lists:** Bullet points are replaced by small "ink splatter" dots or aggressive arrowheads (>).
- **Progress Bars:** Segmented blocks that fill up with Safety Orange, resembling a mechanical gauge.
- **Taped Overlays:** Use a "Washi Tape" visual style for tooltips or temporary notices, appearing as a semi-transparent gray rectangle over the content.
