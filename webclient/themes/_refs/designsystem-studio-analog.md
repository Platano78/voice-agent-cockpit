## Brand & Style

This design system draws direct inspiration from the golden era of high-fidelity audio equipment. It prioritizes the **Tactile / Skeuomorphic** movement, emphasizing physical permanence, machined precision, and the warmth of analog electronics. The target audience values the mechanical ritual of high-end hardware—the weight of a dial, the glow of a vacuum tube, and the clarity of industrial labeling.

The UI should evoke a sense of "heavy" quality. Surfaces are not just flat planes but machined panels with grain and depth. Interactions should feel deliberate, mimicking the physical resistance and "click" of mechanical toggles and weighted aluminum knobs. The emotional response is one of reliability, nostalgia, and premium craftsmanship.

## Layout & Spacing

The layout is a **Fixed Grid** system, mimicking the physical constraints of a rack-mounted unit. Components are grouped into "modules" or "panels" defined by etched lines or physical dividers.

*   **Desktop:** A 12-column grid with generous 32px gutters to prevent the interface from feeling "crowded," maintaining the premium feel of high-end gear.
*   **Mobile:** Content stacks vertically into singular modules. Margins are kept at 24px to ensure the "chassis" of the device is always visible at the edges.
*   **Rhythm:** Spacing follows a strict 8px base unit. Labels should be placed exactly 4px or 8px above their corresponding control (knob/toggle) to ensure functional mapping.

## Elevation & Depth

Depth is the core of this design system. We move away from flat surfaces toward **machined depth**:

1.  **Panel Layers:** Use subtle linear gradients on the `#C0C0C0` surfaces (top-left to bottom-right) to simulate light hitting metal.
2.  **Inset Elements:** Displays and input fields must use `inner-shadows` to appear recessed into the metal faceplate.
3.  **Physical Controls:** Buttons and knobs use complex shadows: a sharp 1px light highlight on the top edge and a soft, dark drop shadow on the bottom to create a "protruding" effect.
4.  **The Glow:** Status lamps and active displays should use a soft Gaussian blur (Bloom) of the primary or secondary color to simulate light leakage from an incandescent bulb.

## Components

*   **Machined Knobs (Sliders/Dials):** Use a circular element with a radial gradient. The "indicator line" should be `#CC5500`. Interaction involves "rotating" the indicator line along a circular track.
*   **Toggle Switches:** Instead of standard checkboxes, use vertical toggle switches. The "Up" position is ON (highlighted with an orange glow lamp above it), and "Down" is OFF.
*   **Status Lamps:** Small circular indicators. When "Off," they are a dark, desaturated version of the color. When "On," they have a bright center and a surrounding glow (bloom).
*   **LCD Displays:** Recessed rectangular areas with a `#1A1A1A` background and `#FFFDD0` or `#CC5500` Space Mono text. Add a very faint scanline overlay to enhance the analog feel.
*   **Buttons:** Rectangular with a "brushed silver" finish. On hover, the button depresses (remove the bottom shadow and add a 1px inner shadow).
*   **VU Meters:** Use for any data visualization. A semi-circular scale with a physical needle that moves dynamically. The "Peak" zone of the meter should be highlighted in burnt orange.
