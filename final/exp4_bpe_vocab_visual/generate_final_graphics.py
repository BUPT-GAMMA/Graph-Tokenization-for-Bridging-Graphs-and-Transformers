import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math
import cairosvg

# --- 配置 ---
PROJECT_ROOT = Path('/home/gzy/py/tokenizerGraph')
VIS_FOLDER = PROJECT_ROOT / 'vocabulary_visualization'
INPUT_CSV_PATH = VIS_FOLDER / 'canonical_fragments.csv'

# --- 输出文件路径 ---
# (Pillow PNG)
OUTPUT_REPRESENTATIVE_PNG = VIS_FOLDER / 'visualization_representative.png'
OUTPUT_PROCESS_PNG = VIS_FOLDER / 'visualization_process_based.png'
# (RDKit Vector)
OUTPUT_REPRESENTATIVE_SVG = VIS_FOLDER / 'visualization_representative.svg'
OUTPUT_REPRESENTATIVE_PDF = VIS_FOLDER / 'visualization_representative.pdf'
OUTPUT_PROCESS_SVG = VIS_FOLDER / 'visualization_process_based.svg'
OUTPUT_PROCESS_PDF = VIS_FOLDER / 'visualization_process_based.pdf'

# --- 图像渲染配置 ---
CELL_W, CELL_H = (340, 300)
FONT_SIZE = 23
PADDING = 20
BACKGROUND_COLOR = "white"
FONT_FAMILY = "DejaVuSans.ttf"
try:
    FONT = ImageFont.truetype(FONT_FAMILY, FONT_SIZE)
except IOError:
    FONT = ImageFont.load_default()

# --- 精选列表 ---
REPRESENTATIVE_SELECTION = { "芳香环: 吡啶": "c1ncccc1", "芳香环: 氯苯": "c1ccc(Cl)cc1", "芳香环: 吲哚": "c1ccc2[nH]ccc2c1", "官能团: 羧酸": "C(=O)O", "官能团: 磺酰胺": "S(=O)(=O)N", "官能团: 三氟甲基": "C(F)(F)F", "脂肪环: 哌啶": "C1CCNCC1", "脂肪环: 吗啉": "C1COCCN1", "复杂官能团: 苯甲酰胺": "NC(=O)c1ccccc1",}
PROCESS_BASED_SELECTION = [ {"group": 1, "step": 1, "smiles": "C(=O)N", "desc": "Amide Base"}, {"group": 1, "step": 2, "smiles": "CC(=O)N", "desc": "Acetamide"}, {"group": 1, "step": 3, "smiles": "NC(=O)C(C)C", "desc": "Isobutyramide"}, {"group": 2, "step": 1, "smiles": "S(=O)(=O)", "desc": "Sulfonyl Base"}, {"group": 2, "step": 2, "smiles": "CS(=O)(=O)", "desc": "Mesyl Group"}, {"group": 2, "step": 3, "smiles": "CCS(=O)(=O)", "desc": "Ethylsulfonyl Group"}, {"group": 3, "step": 1, "smiles": "c1ccccc1", "desc": "Phenyl"}, {"group": 3, "step": 2, "smiles": "Cc1ccccc1", "desc": "Toluene"}, {"group": 3, "step": 3, "smiles": "CCc1ccccc1", "desc": "Ethylbenzene"},]

def canonicalize_smiles(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

# --- 方法1: 使用Pillow进行精细布局 (输出PNG) ---
def render_png_cell(token_id: int, smiles: str) -> Image.Image:
    image = Image.new("RGB", (CELL_W, CELL_H), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol_image = Draw.MolToImage(mol, size=(CELL_W - PADDING * 2, CELL_H - 110))
        x_offset = (CELL_W - mol_image.width) // 2
        y_offset = (CELL_H - mol_image.height - 95) // 2
        image.paste(mol_image, (x_offset, y_offset))
    
    id_text = f"Token ID: {token_id}"
    seq_text = f"Sequence: {smiles}"
    
    def draw_text_centered(text, y_pos):
        bbox = draw.textbbox((0, 0), text, font=FONT)
        x_pos = (CELL_W - (bbox[2] - bbox[0])) / 2
        draw.text((x_pos, y_pos), text, fill="black", font=FONT)

    line_height = FONT_SIZE + 12
    seq_y = CELL_H - PADDING - FONT_SIZE - 5
    id_y = seq_y - line_height
    
    draw_text_centered(seq_text, seq_y)
    draw_text_centered(id_text, id_y)
    return image

def create_png_grid(data, items_per_row, output_path):
    images = [render_png_cell(row['token_id'], row['canonical_smiles']) for _, row in data.iterrows()]
    if not images: return
    
    # 特殊处理过程性示例，确保分组对齐
    if 'group' in data.columns:
        image_groups = []
        grouped = data.groupby('group')
        max_group_size = grouped.size().max()
        for name, group in grouped:
            group_images = [render_png_cell(r['token_id'], r['canonical_smiles']) for _, r in group.iterrows()]
            while len(group_images) < max_group_size:
                group_images.append(Image.new("RGB", (CELL_W, CELL_H), BACKGROUND_COLOR))
            image_groups.extend(group_images)
        images = image_groups

    num_rows = math.ceil(len(images) / items_per_row)
    grid_image = Image.new("RGB", (items_per_row * CELL_W, num_rows * CELL_H), BACKGROUND_COLOR)
    for i, img in enumerate(images):
        row, col = i // items_per_row, i % items_per_row
        grid_image.paste(img, (col * CELL_W, row * CELL_H))
    
    grid_image.save(output_path)
    print(f"🖼️ 精细布局 (PNG) 已保存到: {output_path}")

# --- 方法2: 使用RDKit直接生成 (输出SVG/PDF) ---
def create_vector_grid(data, items_per_row, out_svg, out_pdf):
    mols = [Chem.MolFromSmiles(s) for s in data['canonical_smiles']]
    legends = [f"Token ID: {r['token_id']}\nSequence: {r['canonical_smiles']}" for _, r in data.iterrows()]
    
    draw_options = Draw.MolDrawOptions()
    draw_options.legendFontSize = int(FONT_SIZE * 0.7)
    draw_options.padding = 0.1
    
    svg = Draw.MolsToGridImage(mols, molsPerRow=items_per_row, subImgSize=(CELL_W, CELL_H), legends=legends, useSVG=True, drawOptions=draw_options)
    
    with open(out_svg, 'w') as f: f.write(svg)
    print(f"🎨 矢量图 (SVG) 已保存到: {out_svg}")
    try:
        cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to=str(out_pdf))
        print(f"🎨 矢量图 (PDF) 已保存到: {out_pdf}")
    except Exception as e: print(f"❌ SVG到PDF转换失败: {e}")

def main():
    RDLogger.DisableLog('rdApp.*')
    try: df_canonical = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError: sys.exit(f"错误: 输入文件未找到 {INPUT_CSV_PATH}。")

    # 筛选代表性片段
    rep_data = [df_canonical[df_canonical['canonical_smiles'] == canonicalize_smiles(s)].iloc[0] for r, s in REPRESENTATIVE_SELECTION.items() if not df_canonical[df_canonical['canonical_smiles'] == canonicalize_smiles(s)].empty]
    df_rep = pd.DataFrame(rep_data)
    print(f"--- 已筛选 {len(df_rep)} 个代表性结构 ---")
    
    # 筛选过程性片段
    proc_data = []
    for sel in PROCESS_BASED_SELECTION:
        match = df_canonical[df_canonical['canonical_smiles'] == canonicalize_smiles(sel['smiles'])]
        if not match.empty:
            info = match.iloc[0].to_dict()
            info.update(sel)
            proc_data.append(info)
    df_proc = pd.DataFrame(proc_data).sort_values(by=['group', 'step'])
    print(f"--- 已筛选 {len(df_proc)} 个过程性结构 ---")
    
    # 生成所有图像
    if not df_rep.empty:
        create_png_grid(df_rep, 3, OUTPUT_REPRESENTATIVE_PNG)
        create_vector_grid(df_rep, 3, OUTPUT_REPRESENTATIVE_SVG, OUTPUT_REPRESENTATIVE_PDF)
    
    if not df_proc.empty:
        max_group_size = df_proc.groupby('group').size().max()
        create_png_grid(df_proc, max_group_size, OUTPUT_PROCESS_PNG)
        create_vector_grid(df_proc, max_group_size, OUTPUT_PROCESS_SVG, OUTPUT_PROCESS_PDF)

if __name__ == "__main__":
    main()
