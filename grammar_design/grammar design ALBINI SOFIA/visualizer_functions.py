import os
from lark import tree

def create_pdf_and_svg(parse_tree : tree.Tree, string : str, pdf_folder : str, svg_folder: str):
    graph = tree.pydot__tree_to_graph(parse_tree, "TB")
    if not os.path.exists(pdf_folder):
        os.mkdir(pdf_folder)
    if not os.path.exists(svg_folder):
        os.mkdir(svg_folder)
    graph.write_svg(f"svg/Tree_{string}.svg")
    graph.write_pdf(f"pdf/Tree_{string}.pdf")