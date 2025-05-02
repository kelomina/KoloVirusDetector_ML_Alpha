#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import json
from datetime import datetime

from models import MalwarePredictor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Predictor')

def main():
    parser = argparse.ArgumentParser(description='恶意软件检测预测工具')
    
    # 模型类型选择
    parser.add_argument('--model-type', type=str, choices=['lightgbm', 'deep', 'ensemble'], 
                      default='lightgbm', help='模型类型: lightgbm, deep, ensemble')
    
    # 模型路径参数
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--feature-engineer', type=str, help='特征工程器路径')
    parser.add_argument('--swa-model', type=str, help='SWA模型路径 (仅深度学习和集成模型)')
    parser.add_argument('--lgbm-model', type=str, help='LightGBM模型路径 (仅集成模型)')
    parser.add_argument('--scaler', type=str, help='标准化器路径 (仅深度学习和集成模型)')
    
    # 预测选项
    parser.add_argument('--file', type=str, help='待预测文件路径')
    parser.add_argument('--dir', type=str, help='待预测文件目录')
    parser.add_argument('--recursive', action='store_true', help='递归扫描目录')
    parser.add_argument('--extension', type=str, default='.exe,.dll,.sys', 
                      help='要扫描的文件扩展名，逗号分隔')
    
    # 多模态特征选项
    parser.add_argument('--use-byte-seq', action='store_true', default=True, 
                      help='使用字节序列特征 (仅深度学习和集成模型)')
    parser.add_argument('--use-image', action='store_true', default=True, 
                      help='使用图像特征 (仅深度学习和集成模型)')
    parser.add_argument('--no-byte-seq', action='store_false', dest='use_byte_seq', 
                      help='不使用字节序列特征 (仅深度学习和集成模型)')
    parser.add_argument('--no-image', action='store_false', dest='use_image', 
                      help='不使用图像特征 (仅深度学习和集成模型)')
    
    # 输出选项
    parser.add_argument('--threshold', type=float, help='分类阈值，默认使用模型内部阈值')
    parser.add_argument('--details', action='store_true', help='输出详细预测信息')
    parser.add_argument('--output', type=str, help='结果输出路径')
    parser.add_argument('--json', action='store_true', help='以JSON格式输出结果')
    parser.add_argument('--csv', action='store_true', help='以CSV格式输出结果')
    
    args = parser.parse_args()
    
    # 检查输入参数
    if not args.file and not args.dir:
        parser.error("必须提供 --file 或 --dir 参数")
    
    # 检查模型类型和必要参数
    if args.model_type == 'deep' and not args.swa_model:
        logger.warning("未提供SWA模型路径，性能可能不如最优")
    
    if args.model_type == 'ensemble' and not args.lgbm_model:
        parser.error("集成模型需要提供 --lgbm-model 参数")
    
    # 初始化预测器
    predictor = MalwarePredictor(model_type=args.model_type)
    
    try:
        # 加载模型
        predictor.load_model(
            model_path=args.model,
            feature_engineer_path=args.feature_engineer,
            swa_model_path=args.swa_model,
            lgbm_model_path=args.lgbm_model,
            scaler_path=args.scaler
        )
        
        # 收集待预测文件
        files_to_predict = []
        
        if args.file:
            if os.path.exists(args.file):
                files_to_predict.append(args.file)
            else:
                logger.error(f"文件不存在: {args.file}")
                return 1
        
        if args.dir:
            if not os.path.exists(args.dir):
                logger.error(f"目录不存在: {args.dir}")
                return 1
            
            # 解析扩展名
            extensions = args.extension.split(',')
            
            # 收集符合扩展名的文件
            if args.recursive:
                for root, _, files in os.walk(args.dir):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in extensions):
                            files_to_predict.append(os.path.join(root, file))
            else:
                for file in os.listdir(args.dir):
                    if any(file.lower().endswith(ext) for ext in extensions):
                        files_to_predict.append(os.path.join(args.dir, file))
        
        if not files_to_predict:
            logger.error("没有找到符合条件的文件")
            return 1
        
        logger.info(f"共找到 {len(files_to_predict)} 个文件待预测")
        
        # 执行预测
        if len(files_to_predict) == 1:
            # 单个文件预测
            result = predictor.predict_file(
                file_path=files_to_predict[0],
                use_byte_seq=args.use_byte_seq,
                use_image=args.use_image,
                threshold=args.threshold,
                return_details=args.details
            )
            results = [result]
        else:
            # 批量预测
            results = predictor.predict_batch(
                file_paths=files_to_predict,
                use_byte_seq=args.use_byte_seq,
                use_image=args.use_image,
                threshold=args.threshold,
                return_details=args.details
            )
        
        # 输出结果
        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            if args.json:
                # JSON格式输出
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"结果已保存到: {args.output}")
            elif args.csv:
                # CSV格式输出
                import pandas as pd
                # 将结果转换为扁平的字典列表
                flat_results = []
                for r in results:
                    flat_dict = {}
                    # 处理嵌套结构
                    for k, v in r.items():
                        if isinstance(v, dict):
                            for sub_k, sub_v in v.items():
                                flat_dict[f"{k}_{sub_k}"] = sub_v
                        elif isinstance(v, list):
                            continue  # 跳过列表类型的值
                        else:
                            flat_dict[k] = v
                    flat_results.append(flat_dict)
                    
                # 保存为CSV
                df = pd.DataFrame(flat_results)
                df.to_csv(args.output, index=False, encoding='utf-8')
                logger.info(f"结果已保存到: {args.output}")
            else:
                # 文本格式输出
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(f"恶意软件检测结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"模型类型: {args.model_type}\n")
                    f.write(f"模型路径: {args.model}\n\n")
                    
                    for i, result in enumerate(results):
                        f.write(f"文件 {i+1}: {result['file']}\n")
                        f.write(f"  预测结果: {result['prediction']}\n")
                        f.write(f"  恶意概率: {result['probability']:.4f}\n")
                        
                        # 输出集成模型的分开概率
                        if args.model_type == 'ensemble' and 'lgbm_probability' in result:
                            f.write(f"  LightGBM概率: {result['lgbm_probability']:.4f}\n")
                            f.write(f"  深度学习概率: {result['deep_probability']:.4f}\n")
                        
                        if 'error' in result:
                            f.write(f"  错误: {result['error']}\n")
                            
                        f.write("\n")
                    
                logger.info(f"结果已保存到: {args.output}")
        else:
            # 直接输出到控制台
            print(f"\n恶意软件检测结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"模型类型: {args.model_type}")
            print(f"模型路径: {args.model}\n")
            
            for i, result in enumerate(results):
                print(f"文件 {i+1}: {result['file']}")
                print(f"  预测结果: {result['prediction']}")
                print(f"  恶意概率: {result['probability']:.4f}")
                
                # 输出集成模型的分开概率
                if args.model_type == 'ensemble' and 'lgbm_probability' in result:
                    print(f"  LightGBM概率: {result['lgbm_probability']:.4f}")
                    print(f"  深度学习概率: {result['deep_probability']:.4f}")
                
                if 'error' in result:
                    print(f"  错误: {result['error']}")
                    
                print("")
        
        # 统计结果
        malware_count = sum(1 for r in results if r.get('prediction') == 'Malware')
        benign_count = sum(1 for r in results if r.get('prediction') == 'Benign')
        error_count = sum(1 for r in results if 'error' in r)
        
        logger.info(f"检测结果统计: 总文件数 {len(results)}, 恶意 {malware_count}, 良性 {benign_count}, 错误 {error_count}")
        
        return 0
        
    except Exception as e:
        logger.error(f"预测过程中出错: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 