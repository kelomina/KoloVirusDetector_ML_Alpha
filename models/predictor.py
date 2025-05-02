import os
import logging
import numpy as np
import torch
import joblib
import pickle

from data_processing import load_single_file
from models import MalwareDetector, DeepMalwareDetector, EnsembleModel
from feature_extraction import extract_api_sequence
from feature_extraction.byte_sequence import extract_bytes_from_file
from feature_extraction.image_features import generate_pe_image

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MalwarePredictor')

class MalwarePredictor:
    """
    恶意软件预测器，支持多种模型类型：
    1. LightGBM模型
    2. 深度学习模型
    3. 集成模型

    这个类提供了统一的接口来加载模型和进行预测。
    """
    def __init__(self, model_type='lightgbm'):
        """
        初始化预测器

        Args:
            model_type: 模型类型，可选值为 'lightgbm', 'deep', 'ensemble'
        """
        self.model_type = model_type
        self.model = None
        self.feature_engineer = None
        self.scaler = None
        self.tokenizer = None

    def load_model(self, model_path, feature_engineer_path=None, swa_model_path=None,
                  lgbm_model_path=None, scaler_path=None):
        """
        加载模型和相关组件

        Args:
            model_path: 模型文件路径
            feature_engineer_path: 特征工程器路径
            swa_model_path: SWA模型路径 (仅深度学习模型需要)
            lgbm_model_path: LightGBM模型路径 (仅集成模型需要)
            scaler_path: 标准化器路径 (仅深度学习和集成模型需要)

        Returns:
            self: 返回自身实例以支持链式调用
        """
        try:
            logger.info(f"加载{self.model_type}类型的模型...")
            
            # 加载特征工程器
            if feature_engineer_path:
                logger.info(f"加载特征工程器: {feature_engineer_path}")
                self.feature_engineer = joblib.load(feature_engineer_path)
            
            # 加载标准化器
            if self.model_type in ['deep', 'ensemble']:
                if scaler_path:
                    scaler_file = scaler_path
                elif feature_engineer_path:
                    # 如果未指定scaler_path，尝试在feature_engineer_path同目录下查找
                    scaler_file = os.path.join(os.path.dirname(feature_engineer_path), 'scaler.pkl')
                else:
                    scaler_file = None
                
                if scaler_file and os.path.exists(scaler_file):
                    logger.info(f"加载标准化器: {scaler_file}")
                    self.scaler = joblib.load(scaler_file)
                else:
                    # 如果没有找到标准化器，创建一个无操作标准化器
                    from sklearn.preprocessing import StandardScaler
                    logger.warning("未找到标准化器，使用默认标准化器")
                    self.scaler = StandardScaler()
                    
            # 根据模型类型加载不同的模型
            if self.model_type == 'lightgbm':
                logger.info(f"加载LightGBM模型: {model_path}")
                self.model = MalwareDetector.load_model(model_path)
                
            elif self.model_type == 'deep':
                if not swa_model_path:
                    logger.warning("未提供SWA模型路径，性能可能不如最优")
                
                logger.info(f"加载深度学习模型: {model_path}")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model = DeepMalwareDetector.load_model(model_path, swa_model_path, device)
                
                # 尝试加载tokenizer
                tokenizer_path = os.path.join(os.path.dirname(model_path), 'api_tokenizer.pkl')
                if os.path.exists(tokenizer_path):
                    logger.info(f"加载API tokenizer: {tokenizer_path}")
                    with open(tokenizer_path, 'rb') as f:
                        self.tokenizer = pickle.load(f)
                
            elif self.model_type == 'ensemble':
                if not lgbm_model_path:
                    logger.error("集成模型需要提供LightGBM模型路径")
                    raise ValueError("集成模型需要提供LightGBM模型路径")
                    
                logger.info(f"加载集成模型: {model_path}")
                self.model = EnsembleModel.load(
                    ensemble_path=model_path,
                    deep_model_path=model_path,
                    swa_model_path=swa_model_path,
                    lgbm_model_path=lgbm_model_path,
                    lgbm_feature_engineer_path=feature_engineer_path
                )
            
            logger.info("模型加载完成")
            return self
            
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            raise
            
    def predict_file(self, file_path, use_byte_seq=True, use_image=True, threshold=None, 
                     return_raw_proba=False, return_details=False):
        """
        对单个文件进行预测

        Args:
            file_path: 待预测文件路径
            use_byte_seq: 是否使用字节序列特征 (仅深度学习和集成模型)
            use_image: 是否使用图像特征 (仅深度学习和集成模型)
            threshold: 分类阈值，默认使用模型内部阈值
            return_raw_proba: 是否返回原始概率分数
            return_details: 是否返回详细预测信息

        Returns:
            dict: 预测结果字典
        """
        if not self.model:
            logger.error("模型未加载，请先调用load_model方法")
            raise ValueError("模型未加载，请先调用load_model方法")
            
        try:
            logger.info(f"对文件进行预测: {file_path}")
            
            # 提取静态特征
            logger.info(f"从文件提取特征: {file_path}")
            X = load_single_file(file_path)
            
            if X is None:
                logger.error(f"提取特征失败: {file_path}")
                return {'error': '提取特征失败', 'file': file_path}
            
            # 应用特征工程
            if self.feature_engineer:
                X_fe = self.feature_engineer.transform(X)
            else:
                X_fe = X
                
            # LightGBM模型预测
            if self.model_type == 'lightgbm':
                # 预测
                pred, prob = self.model.predict(X_fe, threshold)
                
                result = {
                    'file': file_path,
                    'prediction': 'Malware' if pred[0] else 'Benign',
                    'probability': float(prob[0])
                }
                
            # 深度学习模型预测
            elif self.model_type in ['deep', 'ensemble']:
                # 应用标准化
                if self.scaler:
                    X_scaled = self.scaler.transform(X_fe)
                else:
                    X_scaled = X_fe
                
                # 提取API序列
                logger.info(f"提取API序列: {file_path}")
                api_sequence = extract_api_sequence(file_path)
                
                # 提取字节序列（如果启用）
                byte_sequence = None
                if use_byte_seq:
                    logger.info(f"提取字节序列: {file_path}")
                    byte_sequence = extract_bytes_from_file(file_path)
                    byte_sequence = np.expand_dims(byte_sequence, axis=0)  # 添加batch维度
                
                # 提取图像特征（如果启用）
                image = None
                if use_image:
                    logger.info(f"生成图像特征: {file_path}")
                    image = generate_pe_image(file_path)
                    image = np.expand_dims(image, axis=0)  # 添加batch维度
                
                # 根据模型类型进行预测
                if self.model_type == 'deep':
                    # 深度学习模型预测
                    pred, prob = self.model.predict(
                        pe_features=X_scaled,
                        api_sequences=[api_sequence],
                        use_swa=True,
                        byte_sequences=byte_sequence,
                        images=image
                    )
                    
                    result = {
                        'file': file_path,
                        'prediction': 'Malware' if pred[0] else 'Benign',
                        'probability': float(prob[0])
                    }
                    
                elif self.model_type == 'ensemble':
                    # 集成模型预测
                    pred, prob, separate_probs = self.model.predict(
                        pe_features=X_scaled,
                        api_sequences=[api_sequence],
                        byte_sequences=byte_sequence,
                        images=image,
                        use_deep_swa=True,
                        return_separate_probs=True
                    )
                    
                    result = {
                        'file': file_path,
                        'prediction': 'Malware' if pred[0] else 'Benign',
                        'probability': float(prob[0]),
                        'lgbm_probability': float(separate_probs[0][0]),
                        'deep_probability': float(separate_probs[1][0])
                    }
            
            logger.info(f"预测结果: {result}")
            
            # 添加原始概率分数
            if return_raw_proba:
                result['raw_probability'] = float(prob[0])
                
            # 添加详细信息
            if return_details:
                # 提取文件基本信息
                from data_processing.checkpoint_extractor import extract_file_info
                file_info = extract_file_info(file_path)
                if file_info:
                    result['file_info'] = file_info
                    
                # 添加主要特征及其重要性
                if self.model_type == 'lightgbm' and hasattr(self.model, 'feature_importance'):
                    # 获取Top 10重要特征
                    top_features = self.model.feature_importance.head(10)
                    result['important_features'] = top_features.to_dict('records')
                
            return result
                
        except Exception as e:
            logger.error(f"预测文件时出错: {str(e)}")
            return {'error': str(e), 'file': file_path}
    
    def predict_batch(self, file_paths, use_byte_seq=True, use_image=True, threshold=None,
                     return_raw_proba=False, return_details=False):
        """
        批量预测多个文件

        Args:
            file_paths: 文件路径列表
            use_byte_seq: 是否使用字节序列特征 (仅深度学习和集成模型)
            use_image: 是否使用图像特征 (仅深度学习和集成模型)
            threshold: 分类阈值，默认使用模型内部阈值
            return_raw_proba: 是否返回原始概率分数
            return_details: 是否返回详细预测信息

        Returns:
            list: 预测结果列表
        """
        results = []
        for file_path in file_paths:
            result = self.predict_file(
                file_path, 
                use_byte_seq=use_byte_seq, 
                use_image=use_image,
                threshold=threshold,
                return_raw_proba=return_raw_proba,
                return_details=return_details
            )
            results.append(result)
        
        return results 