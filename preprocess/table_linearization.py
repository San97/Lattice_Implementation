from preprocess_utils import get_highlighted_subtable


def table_do_linearize(input_table, indices, name, section_name, ordered=False):
    # Initialize the table string, type, row and column IDs.
    linearized_table_str = ""
    id_of_type = []
    id_of_row = []
    id_of_col = []
    # If the page title is not None, add it to the table string.
    if name:
        linearized_table_str += "<page_title> " + name + " </page_title> "
        id_of_type += [1 for _ in range(len(linearized_table_str))]

    # If the section title is not None, add it to the table string.
    if section_name:
        linearized_table_str += "<section_title> " + section_name + " </section_title> "
        id_of_type += [2 for _ in range(len(linearized_table_str) - len(id_of_type))]

    # Initialize the type, row and column IDs for the table string.
    id_of_row += [0 for _ in range(len(id_of_type))]
    id_of_col += [0 for _ in range(len(id_of_type))]

    # Add the opening tag for the table.
    linearized_table_str += "<table> "
    id_of_type += [3 for _ in range(len("<table> "))]
    id_of_row += [0 for _ in range(len("<table> "))]
    id_of_col += [0 for _ in range(len("<table> "))]

    # Get the highlighted subtable.
    subtable = get_highlighted_subtable(input_table, indices, with_heuristic_headers=True)

    # Sort the subtable items by cell value if order_cell is True.
    if ordered:
        subtable = sorted(subtable, key=lambda x: x["cell"]["value"])

    # Iterate over the subtable items.
    for item in subtable:
        # Get the cell, row headers, column headers, row index, and column index.
        cell = item["cell"]
        row_headers = item["row_headers"]
        col_headers = item["col_headers"]
        r_index = item["row_index"]
        c_index = item["col_index"]

        # Initialize the cell string.
        item_str = "<cell> " + cell["value"] + " "

        # Add the header tags for each header associated with the cell.
        headers = col_headers + row_headers
        headers = sorted(headers, key=lambda x: x["value"])
        for header in headers:
            item_str += "<header> " + header["value"] + " </header> "

        # Add the closing tag for the cell.
        item_str += "</cell> "
        cell_length = len(item_str)

        # Append the cell string to the table string and update the type, row, and column IDs.
        linearized_table_str += item_str
        id_of_type += [3 for _ in range(cell_length)]
        id_of_row += [r_index + 1 for _ in range(cell_length)]
        id_of_col += [c_index + 1 for _ in range(cell_length)]

    # Add the closing tag for the table.
    linearized_table_str += "</table>"
    id_of_type += [3 for _ in range(len("</table>"))]
    id_of_row += [0 for _ in range(len("</table>"))]
    id_of_col += [0 for _ in range(len("</table>"))]

    # Check if cell_indices is not empty.
    if indices:
        assert "<cell>" in linearized_table_str

