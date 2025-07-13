import torch
import logging

# 假设 VLKD 是你的模型类
import VLKD  # 替换为实际的模型文件和类名
import config  # 替换为实际的配置文件


# 初始化日志记录器
def logger():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


def log_info(logger, message):
    logger.info(message)


def main():
    log = logger()
    log_info(log, config)

    Model = VLKD(log, config)

    try:
        Model.load_checkpoints(config.CHECKPOINT)
        log_info(log, "Model checkpoint loaded successfully.")
        Model.eval()

        # 进行简单的验证，例如检查模型参数或进行一次前向传播
        dummy_input = torch.randn(1, 3, 224, 224)  # 假设输入是一个 224x224 的 RGB 图像
        with torch.no_grad():
            output = Model(dummy_input)
        log_info(log, f"Model output: {output}")

    except FileNotFoundError:
        log_info(log, "No checkpoint found. Unable to load model.")
    except Exception as e:
        log_info(log, f"Error loading checkpoint: {e}")


if __name__ == "__main__":
    main()

