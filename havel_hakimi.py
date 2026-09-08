"""
Havel-Hakimi Algorithm
Degree-sequence graph construction with step-by-step verification and visualization.
"""

import ast
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================
# DEGREE SEQUENCE - Change here, or leave None for user input
# ============================================================
DEFAULT_SEQUENCE = None  # e.g. [3, 3, 3, 3, 3, 3]


def get_input_sequence():
    """Retrieve the degree sequence from default or user prompt."""
    if DEFAULT_SEQUENCE is not None:
        print(f"  Using default sequence: {DEFAULT_SEQUENCE}")
        return list(DEFAULT_SEQUENCE)

    raw = input("  Enter degree sequence (e.g. [3, 3, 3, 3, 3, 3]): ").strip()
    try:
        seq = ast.literal_eval(raw)
        if not isinstance(seq, list) or len(seq) == 0:
            raise ValueError("Input must be a non-empty list.")
        return [int(x) for x in seq]
    except Exception as e:
        raise SystemExit(f"  ERROR: {e}")


def validate_sequence(sequence):
    """Validate input degree sequence. Returns list of error strings."""
    n = len(sequence)
    errors = []
    for i, d in enumerate(sequence):
        if not isinstance(d, int):
            errors.append(f"Position {i + 1}: not an integer ({d})")
        else:
            if d < 0:
                errors.append(f"Position {i + 1}: negative degree ({d})")
            if d > n - 1:
                errors.append(
                    f"Position {i + 1}: degree {d} exceeds n-1 = {n - 1}"
                )
    if sum(sequence) % 2 != 0:
        errors.append(f"Sum of degrees is odd ({sum(sequence)})")
    return errors


def havel_hakimi(original_sequence):
    """
    Execute the Havel-Hakimi algorithm with vertex identity tracking.

    At each step the vertex with the largest remaining degree is selected,
    removed, and connected to the next d vertices with the highest degrees.
    If at any point the reduction is impossible the sequence is non-graphical.

    Returns (success, steps, edges, failure_info).
    """
    n = len(original_sequence)
    vertices = [(f"V{i + 1}", original_sequence[i]) for i in range(n)]
    steps = []
    edges = []
    step_num = 0

    while True:
        vertices.sort(key=lambda x: (-x[1], x[0]))

        if all(d == 0 for _, d in vertices):
            return True, steps, edges, None

        step_num += 1
        target_label, target_degree = vertices[0]
        current_state = list(vertices)

        remaining = vertices[1:]

        if target_degree > len(remaining):
            return False, steps, edges, {
                "step": step_num,
                "vertex": target_label,
                "required": target_degree,
                "available": len(remaining),
                "remaining_sequence": list(vertices),
            }

        connected_to = []
        for i in range(1, target_degree + 1):
            other_label, other_degree = vertices[i]
            if other_degree <= 0:
                return False, steps, edges, {
                    "step": step_num,
                    "vertex": target_label,
                    "required": target_degree,
                    "available": i - 1,
                    "remaining_sequence": list(vertices),
                    "failed_on": other_label,
                }
            edges.append((target_label, other_label))
            connected_to.append(other_label)
            vertices[i] = (other_label, other_degree - 1)

        vertices.pop(0)
        vertices.sort(key=lambda x: (-x[1], x[0]))

        steps.append({
            "step": step_num,
            "current": current_state,
            "selected": target_label,
            "selected_degree": target_degree,
            "connections": connected_to,
            "reduced": list(vertices),
        })

    return True, steps, edges, None


def verify_graph(edges, original_sequence):
    """Compare actual degrees from constructed edges against the original sequence."""
    n = len(original_sequence)
    labels = [f"V{i + 1}" for i in range(n)]
    actual = {v: 0 for v in labels}
    for v1, v2 in edges:
        actual[v1] += 1
        actual[v2] += 1

    results = []
    all_ok = True
    for i, label in enumerate(labels):
        req = original_sequence[i]
        act = actual[label]
        ok = req == act
        if not ok:
            all_ok = False
        results.append((label, req, act, ok))
    return results, all_ok


def print_step(step_info):
    """Print a single Havel-Hakimi step in the required format."""
    print(f"\nSTEP {step_info['step']}")

    current = step_info["current"]
    degs = ", ".join(str(d) for _, d in current)
    print(f"  Current sequence: [{degs}]")
    mapping = ", ".join(f"{l}={d}" for l, d in current)
    print(f"    ({mapping})")

    print(
        f"  Selected vertex:  {step_info['selected']}"
        f" (degree {step_info['selected_degree']})"
    )
    conns = ", ".join(step_info["connections"])
    print(f"  Connections:      {step_info['selected']} -> {conns}")

    reduced = step_info["reduced"]
    if reduced:
        red_degs = ", ".join(str(d) for _, d in reduced)
        print(f"  Reduced sequence: [{red_degs}]")
        red_map = ", ".join(f"{l}={d}" for l, d in reduced)
        print(f"    ({red_map})")
    else:
        print("  Reduced sequence: []")


def compute_node_params(n):
    """Scale node size, font size, and edge width based on vertex count."""
    if n <= 5:
        return 800, 11, 2.0
    elif n <= 10:
        return 600, 10, 1.6
    elif n <= 20:
        return 400, 8, 1.2
    elif n <= 40:
        return 250, 7, 0.9
    else:
        return 150, 6, 0.6


def visualize_graph(
    edges, original_sequence, success, steps=None, failure_info=None
):
    """Render a clean, deterministic graph visualization."""
    n = len(original_sequence)
    labels = [f"V{i + 1}" for i in range(n)]

    G = nx.Graph()
    G.add_nodes_from(labels)
    G.add_edges_from(edges)

    # Deterministic layout selection
    if n <= 7:
        pos = nx.circular_layout(G)
    elif n <= 15:
        deg_map = dict(G.degree())
        shells = []
        for d in range(max(deg_map.values(), default=0), -1, -1):
            shell = [v for v in labels if deg_map.get(v, 0) == d]
            if shell:
                shells.append(shell)
        if len(shells) > 1:
            pos = nx.shell_layout(G, nlist=shells)
        else:
            pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=1.5 / (n**0.5), iterations=50)

    node_size, font_size, edge_w = compute_node_params(n)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")

    # Identify problematic vertices for failed constructions
    problematic = set()
    if failure_info and failure_info.get("vertex"):
        problematic.add(failure_info["vertex"])

    normal = [v for v in labels if v not in problematic and v in G]
    problems = [v for v in labels if v in problematic and v in G]

    if edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax, width=edge_w, edge_color="#666666", alpha=0.8
        )

    if normal:
        nx.draw_networkx_nodes(
            G, pos, nodelist=normal, ax=ax, node_color="#4A7FB5",
            node_size=node_size, edgecolors="#2C5273", linewidths=1.5,
        )

    if problems:
        nx.draw_networkx_nodes(
            G, pos, nodelist=problems, ax=ax, node_color="#C04040",
            node_size=int(node_size * 1.1), edgecolors="#7A2020",
            linewidths=2.0,
        )

    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=font_size, font_weight="bold",
        font_color="white",
    )

    # Title
    seq_str = str(original_sequence)
    if success:
        title = (
            f"Havel-Hakimi \u2014 Graph Construction\n"
            f"Degree Sequence: {seq_str}"
        )
    else:
        title = (
            f"Havel-Hakimi \u2014 Failed Construction\n"
            f"Degree Sequence: {seq_str}"
        )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    # Information box
    info_lines = [
        f"Vertices: {n}",
        f"Edges: {len(edges)}",
        f"Degree Sequence: {seq_str}",
    ]
    if success:
        info_lines.append("Status: GRAPHICAL")
        if steps:
            info_lines.append(f"Steps: {len(steps)}")
        info_lines.append("Verification: PASSED")
    else:
        info_lines.append("Status: NOT GRAPHICAL")
        if failure_info:
            info_lines.append(f"Failed at Step: {failure_info['step']}")
            rem = [d for _, d in failure_info.get("remaining_sequence", [])]
            info_lines.append(f"Remaining: {rem}")

    ax.text(
        0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes,
        fontsize=8, verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                  edgecolor="#AAAAAA"),
    )

    # Legend for failed constructions
    if not success:
        legend_items = [
            mpatches.Patch(
                facecolor="#4A7FB5", edgecolor="#2C5273",
                label="Processed vertex",
            ),
            mpatches.Patch(
                facecolor="#C04040", edgecolor="#7A2020",
                label="Problematic vertex",
            ),
            plt.Line2D(
                [0], [0], color="#666666", linewidth=1.5,
                label="Established edge",
            ),
        ]
        ax.legend(
            handles=legend_items, loc="lower right", fontsize=7,
            framealpha=0.9, edgecolor="#CCCCCC",
        )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig("graph_output.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.show()


def main():
    print("=" * 52)
    print("          HAVEL-HAKIMI ALGORITHM")
    print("=" * 52)
    print()

    sequence = get_input_sequence()

    print(f"\n  Input Degree Sequence: {sequence}")
    print(f"  Number of vertices:   {len(sequence)}")

    # --- Validation ---
    errors = validate_sequence(sequence)
    if errors:
        print("\n  VALIDATION FAILED:")
        for err in errors:
            print(f"    - {err}")
        return

    # --- Havel-Hakimi reduction ---
    success, steps, edges, failure_info = havel_hakimi(sequence)

    for step in steps:
        print_step(step)

    # --- Result ---
    print("\n" + "=" * 52)
    print("          RESULT")
    print("=" * 52)

    if success:
        print("\n  GRAPHICAL")

        print("\n" + "-" * 52)
        print("  EDGE LIST")
        print("-" * 52)
        for v1, v2 in sorted(edges):
            print(f"  {v1} -- {v2}")

        results, all_ok = verify_graph(edges, sequence)

        print("\n" + "-" * 52)
        print("  DEGREE VERIFICATION")
        print("-" * 52)
        print(f"  {'Vertex':<8} {'Required':<10} {'Actual':<10} {'Status':<8}")
        print(f"  {'------':<8} {'--------':10} {'------':<10} {'------':<8}")
        for label, req, act, ok in results:
            status = "OK" if ok else "MISMATCH"
            print(f"  {label:<8} {req:<10} {act:<10} {status:<8}")
        print(f"\n  GRAPH VERIFIED: {'YES' if all_ok else 'NO'}")

        visualize_graph(edges, sequence, success=True, steps=steps)

    else:
        print("\n  NOT GRAPHICAL")

        if failure_info:
            v = failure_info["vertex"]
            r = failure_info["required"]
            a = failure_info["available"]
            s = failure_info["step"]
            print(f"\n  Havel-Hakimi failed at Step {s}.")
            print(f"  Vertex {v} requires {r} connections,")
            print(f"  but only {a} eligible vertices remain.")
            rem = [d for _, d in failure_info.get("remaining_sequence", [])]
            print(f"  Remaining sequence: {rem}")

        print("\n  NO GRAPH EXISTS for the supplied degree sequence.")

        visualize_graph(
            edges, sequence, success=False, steps=steps,
            failure_info=failure_info,
        )


if __name__ == "__main__":
    main()
