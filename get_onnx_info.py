import onnxruntime as ort

if __name__ == '__main__':

    print(ort.get_available_providers())

    # session = ort.InferenceSession(r"C:\Users\lsn\Downloads\sam3_vit_h\sam3_decoder.onnx")
    #
    # # 查看输入
    # print("【输入信息】")
    # for input in session.get_inputs():
    #     print(f"  名称: {input.name}")
    #     print(f"  形状: {input.shape}")
    #     print(f"  类型: {input.type}")
    #     print()
    #
    # # 查看输出
    # print("【输出信息】")
    # for output in session.get_outputs():
    #     print(f"  名称: {output.name}")
    #     print(f"  形状: {output.shape}")
    #     print(f"  类型: {output.type}")
    #     print()
