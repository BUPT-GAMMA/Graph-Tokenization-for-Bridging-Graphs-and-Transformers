import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont
import math
import csv
from rdkit import RDLogger

# --- 配置 ---
# 设置项目根目录，以便能够导入项目中的模块（如果需要）
PROJECT_ROOT = Path('/home/gzy/py/tokenizerGraph')  # 使用绝对路径确保正确性
sys.path.append(str(PROJECT_ROOT))
CODEBOOK_PATH = PROJECT_ROOT / 'model/bpe/zinc/smiles/multi_100/bpe_codebook.pkl'
OUTPUT_IMAGE_PATH = PROJECT_ROOT / 'learned_vocabulary_zinc_smiles.png'
OUTPUT_CSV_PATH = PROJECT_ROOT / 'valid_smiles_fragments.csv'
NUM_TOKENS_TO_VISUALIZE = 25  # 要可视化的token数量
TOKENS_PER_ROW = 5  # 每行显示的token数量

# --- 图像渲染配置 ---
IMG_SIZE = (300, 300)
FONT_SIZE = 24
PADDING = 20
BACKGROUND_COLOR = "white"

try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
except IOError:
    print("警告: 未找到DejaVuSans字体，将使用默认字体。")
    FONT = ImageFont.load_default()

def load_bpe_codebook(path: Path) -> Dict[str, Any]:
    """加载BPE codebook文件。"""
    if not path.exists():
        raise FileNotFoundError(f"错误: BPE codebook文件未找到: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"✅ BPE codebook加载成功: {path}")
    return data

class BpeDecoder:
    """
    一个解码器，用于将BPE合并的token ID递归地解码回其原始的基础token序列。
    """
    def __init__(self, merge_rules: List[Tuple[int, int, int]]):
        self._build_decoding_maps(merge_rules)

    def _build_decoding_maps(self, merge_rules: List[Tuple[int, int, int]]):
        """构建解码所需的映射表。"""
        self.id_to_pair: Dict[int, Tuple[int, int]] = {new_id: (r1, r2) for r1, r2, new_id in merge_rules}
        all_ids = set()
        new_ids = {rule[2] for rule in merge_rules}
        for r1, r2, _ in merge_rules:
            all_ids.add(r1)
            all_ids.add(r2)
        
        # 基础token是那些从未被创建（即从未作为new_id出现）的token
        self.base_token_ids = all_ids - new_ids
        print(f"🔍 解码器初始化: {len(self.base_token_ids)}个基础token, {len(self.id_to_pair)}条合并规则。")

    def decode_token_to_ids(self, token_id: int) -> List[int]:
        """将单个token ID递归解码为基础token ID的序列。"""
        if token_id in self.base_token_ids:
            return [token_id]
        
        if token_id not in self.id_to_pair:
            # 对于SMILES的ord()编码，所有ID都应该是可解码的。
            # 如果出现这种情况，说明codebook可能有问题。
            print(f"警告: 发现未知token ID: {token_id}，无法解码。")
            return []

        r1, r2 = self.id_to_pair[token_id]
        return self.decode_token_to_ids(r1) + self.decode_token_to_ids(r2)

    def decode_to_string(self, token_id: int) -> str:
        """将token ID完全解码为SMILES字符串片段。"""
        base_ids = self.decode_token_to_ids(token_id)
        # 关键修正：使用chr()将ASCII序数转换回字符
        try:
            chars = [chr(i) for i in base_ids]
            return "".join(chars)
        except ValueError as e:
            print(f"警告: 解码token ID序列 {base_ids} 时出错: {e}")
            return ""


def render_token_image(token_id: int, smiles_fragment: str) -> Image.Image:
    """为单个token及其对应的SMILES片段生成图像。"""
    # 创建一张白色背景的画布
    image = Image.new("RGB", IMG_SIZE, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)

    # 渲染化学结构
    mol = Chem.MolFromSmiles(smiles_fragment, sanitize=True) # 确保进行化学检查
    mol_image = None
    if mol:
        try:
            mol_image = Draw.MolToImage(mol, size=(IMG_SIZE[0] - 2 * PADDING, IMG_SIZE[1] - 60))
        except Exception as e:
            print(f"警告: RDKit无法为SMILES '{smiles_fragment}' 生成图像: {e}")
    
    if mol_image:
        # 将分子图像粘贴到画布中央
        x_offset = (IMG_SIZE[0] - mol_image.width) // 2
        y_offset = (IMG_SIZE[1] - mol_image.height - 40) // 2
        image.paste(mol_image, (x_offset, y_offset))

    # 在图像下方添加文本
    id_text = f"Token ID: {token_id}"
    smiles_text = f"SMILES: {smiles_fragment}"
    
    # 计算文本位置
    id_text_bbox = draw.textbbox((0, 0), id_text, font=FONT)
    smiles_text_bbox = draw.textbbox((0, 0), smiles_text, font=FONT)
    
    id_text_x = (IMG_SIZE[0] - (id_text_bbox[2] - id_text_bbox[0])) / 2
    id_text_y = IMG_SIZE[1] - PADDING - (smiles_text_bbox[3] - smiles_text_bbox[1]) - (id_text_bbox[3] - id_text_bbox[0]) - 5
    
    smiles_text_x = (IMG_SIZE[0] - (smiles_text_bbox[2] - smiles_text_bbox[0])) / 2
    smiles_text_y = IMG_SIZE[1] - PADDING - (smiles_text_bbox[3] - smiles_text_bbox[1])

    draw.text((id_text_x, id_text_y), id_text, fill="black", font=FONT)
    draw.text((smiles_text_x, smiles_text_y), smiles_text, fill="black", font=FONT)

    return image

def main():
    """主函数，执行整个可视化流程。"""
    # 抑制RDKit的错误日志，避免控制台刷屏
    RDLogger.DisableLog('rdApp.*')

    # 1. 加载BPE codebook
    codebook = load_bpe_codebook(CODEBOOK_PATH)
    merge_rules = codebook.get('merge_rules')
    if not merge_rules:
        print("错误: 'merge_rules' 在codebook中未找到或为空。")
        return

    # 2. 初始化解码器
    decoder = BpeDecoder(merge_rules)

    # 3. 全面搜索：遍历所有合并规则，找到所有可解析的SMILES片段
    all_valid_tokens = []
    print("\n--- 开始全面解码并筛选所有可解析的SMILES片段 ---")
    
    # 我们从高频（rank高）到低频（rank低）遍历
    for i, rule in enumerate(reversed(merge_rules)):
        token_id = rule[2]
        merge_rank = len(merge_rules) - 1 - i # 排名越高越重要
        
        smiles_fragment = decoder.decode_to_string(token_id)
        if not smiles_fragment:
            continue

        # 关键步骤：检查SMILES片段是否可以被RDKit解析
        mol = Chem.MolFromSmiles(smiles_fragment, sanitize=True)
        if mol:
            print(f"✅ [Rank {merge_rank}] Token {token_id} -> '{smiles_fragment}' (可解析)")
            all_valid_tokens.append({
                'token_id': token_id,
                'merge_rank': merge_rank,
                'smiles': smiles_fragment
            })

    if not all_valid_tokens:
        print("错误: 在整个codebook中没有找到任何可解析的SMILES片段。")
        return

    print(f"\n--- 搜索完成，共找到 {len(all_valid_tokens)} 个可解析的SMILES片段 ---")

    # 4. 将所有找到的有效token保存到CSV文件
    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['token_id', 'merge_rank', 'smiles'])
        writer.writeheader()
        writer.writerows(all_valid_tokens)
    print(f"💾 所有有效片段已保存到: {OUTPUT_CSV_PATH}")

    # 5. 从所有有效片段中选择排名最高的N个进行可视化
    tokens_for_visualization = all_valid_tokens[:NUM_TOKENS_TO_VISUALIZE]

    # 6. 生成所有单个图像
    images = []
    print(f"\n--- 开始为排名最高的 {len(tokens_for_visualization)} 个片段生成图像 ---")
    for token_info in tokens_for_visualization:
        img = render_token_image(token_info['token_id'], token_info['smiles'])
        images.append(img)
    
    if not images:
        print("错误: 没有生成任何图像。")
        return

    # 7. 将所有图像拼接成一个网格图
    num_rows = math.ceil(len(images) / TOKENS_PER_ROW)
    grid_width = TOKENS_PER_ROW * IMG_SIZE[0]
    grid_height = num_rows * IMG_SIZE[1]
    
    grid_image = Image.new("RGB", (grid_width, grid_height), BACKGROUND_COLOR)
    
    for i, img in enumerate(images):
        row = i // TOKENS_PER_ROW
        col = i % TOKENS_PER_ROW
        x_pos = col * IMG_SIZE[0]
        y_pos = row * IMG_SIZE[1]
        grid_image.paste(img, (x_pos, y_pos))

    # 8. 保存最终的网格图
    grid_image.save(OUTPUT_IMAGE_PATH)
    print(f"🎉 可视化图像成功保存到: {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()
