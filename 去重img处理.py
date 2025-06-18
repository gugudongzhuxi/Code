import os
import shutil
from PIL import Image
import imagehash


def deduplicate_images_by_phash(root, output_dir='deduplicated_images', hash_size=8, threshold=5):
    """
    从 root 目录（含子目录）收集所有图片，使用感知哈希去重，并将唯一图片移动到 output_dir 文件夹。
    报告所有截断的图像但不处理它们。

    参数:
        root (str): 要遍历的根目录路径。
        output_dir (str): 保存去重后图片的输出目录。
        hash_size (int): phash 哈希大小（默认 8）。
        threshold (int): 哈希差异容忍度，小于该值则认为是重复图片。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_hashes = []
    moved_count = 0
    truncated_count = 0
    error_count = 0
    total_images = 0

    # 创建错误日志文件
    error_log = os.path.join(output_dir, "truncated_images.txt")

    with open(error_log, "w") as log_file:
        log_file.write("截断图像列表\n")
        log_file.write("=" * 50 + "\n\n")

        for dirpath, dirnames, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower().endswith(('.jpg', '.jpeg')):
                    total_images += 1
                    fpath = os.path.join(dirpath, fname)
                    try:
                        img = Image.open(fpath).convert("RGB")
                        # 尝试加载图像像素数据，这会触发截断图像的异常
                        img.load()

                        phash = imagehash.phash(img, hash_size=hash_size)

                        is_duplicate = False
                        for existing_hash, _ in image_hashes:
                            if phash - existing_hash < threshold:
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            image_hashes.append((phash, fpath))
                            new_path = os.path.join(output_dir, os.path.basename(fpath))

                            # 如果文件名已存在，添加唯一标识符
                            counter = 1
                            base_name, ext = os.path.splitext(os.path.basename(fpath))
                            while os.path.exists(new_path):
                                new_path = os.path.join(output_dir, f"{base_name}_{counter}{ext}")
                                counter += 1

                            shutil.copy2(fpath, new_path)
                            moved_count += 1
                            if moved_count % 100 == 0:
                                print(f"已处理 {moved_count} 张唯一图片 (总共扫描了 {total_images} 张图片)")

                    except OSError as e:
                        if "truncated" in str(e).lower():
                            truncated_count += 1
                            error_msg = f"❌ Error processing {fpath}: {e}"
                            print(error_msg)
                            log_file.write(f"{error_msg}\n")
                        else:
                            error_count += 1
                            print(f"❌ 其他错误 {fpath}: {e}")

                    except Exception as e:
                        error_count += 1
                        print(f"❌ 未知错误 {fpath}: {e}")

        # 写入统计信息到日志
        summary = f"""
处理统计
=============================
总扫描图像: {total_images}
唯一图像: {moved_count}
截断的图像: {truncated_count}
其他错误: {error_count}
"""
        log_file.write("\n" + summary)

    print(f"\n✅ 去重完成，共保存了 {moved_count} 张唯一图片到 '{output_dir}'")
    print(f"总扫描图像: {total_images}, 截断的图像: {truncated_count}, 其他错误: {error_count}")
    print(f"截断图像列表已保存到: {error_log}")


if __name__ == "__main__":
    root = "/industai_data/无人机/无人机数据集/2025-04-23"
    output_dir = "/industai_data/无人机/output/2025-04-23"
    deduplicate_images_by_phash(root, output_dir)