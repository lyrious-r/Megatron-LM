import pickle
from pprint import pformat

# TODO: 替换为你定义的类名
class YourClassName:
    def __init__(self, _timing_data, _stored_activation_data, _peak_activation_data,
                 _model_state_data, _max_throughput_achieved):
        self.timing_data = _timing_data
        self.stored_activation_data = _stored_activation_data
        self.peak_activation_data = _peak_activation_data
        self.model_state_data = _model_state_data
        self.max_throughput_achieved = _max_throughput_achieved

    @classmethod
    def deserialize(cls, serialized):
        (
            timing_data,
            stored_activation_data,
            peak_activation_data,
            model_state_data,
            max_throughput_achieved,
        ) = pickle.loads(serialized)
        return cls(
            _timing_data=timing_data,
            _stored_activation_data=stored_activation_data,
            _peak_activation_data=peak_activation_data,
            _model_state_data=model_state_data,
            _max_throughput_achieved=max_throughput_achieved,
        )

def deserialize_and_save(pkl_path, txt_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    output_lines = []
    for key, serialized in data.items():
        try:
            obj = YourClassName.deserialize(serialized)
            output_lines.append(f"Key: {key}\n")
            output_lines.append("timing_data:\n" + pformat(obj.timing_data, indent=4) + "\n")
            output_lines.append("stored_activation_data:\n" + pformat(obj.stored_activation_data, indent=4) + "\n")
            output_lines.append("peak_activation_data:\n" + pformat(obj.peak_activation_data, indent=4) + "\n")
            output_lines.append("model_state_data:\n" + pformat(obj.model_state_data, indent=4) + "\n")
            output_lines.append("max_throughput_achieved:\n" + pformat(obj.max_throughput_achieved, indent=4) + "\n")
            output_lines.append("=" * 80 + "\n")
        except Exception as e:
            output_lines.append(f"Key: {key}\nFailed to deserialize: {e}\n")
            output_lines.append("=" * 80 + "\n")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)

    print(f"解析完成，写入：{txt_path}")

# 示例调用
if __name__ == "__main__":
    deserialize_and_save("cost_models/gpt_6.7b_cm.pkl", "printpkl/gpt_6.7b_cm.txt")

