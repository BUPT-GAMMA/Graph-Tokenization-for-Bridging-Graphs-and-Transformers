from __future__ import annotations


def load_pretrained_backbone(config):
    """加载预训练 BERT（MLM）作为微调 backbone。"""
    from src.models.bert.model import BertMLM
    pretrained_dir = config.get_bert_model_path("pretrained").parent
    return BertMLM.load_model(str(pretrained_dir))


def build_task_model(
    config,
    pretrained,
    udi,
    method,
):
    """
    构建统一任务模型，返回模型和任务处理器
    
    Args:
        config: 项目配置
        pretrained: 预训练BERT模型
        udi: 统一数据接口
        method: 序列化方法
        
    Returns:
        (model, task_handler) 元组
    """
    from src.models.bert.heads import create_model_from_udi
    
    pooling_method = config.bert.architecture.pooling_method
    model, task_handler = create_model_from_udi(udi, pretrained, pooling_method)
    
    return model, task_handler





