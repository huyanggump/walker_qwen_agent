# coding: utf-8
# @Author: WalkerZ
# @Time: 2024/10/16

from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
from dateutil import parser
import json
import torch
import datetime
import re
from config import model_file,api_log_file
from caldav import DAVClient
from icalendar import Calendar, Event
import logging
from transformers import MllamaForConditionalGeneration, AutoProcessor

# 配置logging模块
logging.basicConfig(
    filename=api_log_file,  # 日志文件名
    filemode='a',             # 追加模式
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    level=logging.INFO        # 日志级别
)

# Flask 初始化
app = Flask(__name__)

# 强制模型仅在GPU上运行
if torch.cuda.is_available():
    device = "cuda"  # 指定仅使用CUDA设备
else:
    raise RuntimeError("No GPU available. Please ensure a GPU is available for model execution.")


# 加载模型
tokenizer = AutoTokenizer.from_pretrained(model_file)
# model = AutoModelForCausalLM.from_pretrained(model_file, torch_dtype=torch.float16, device_map={"": device}) # .to("cuda")


model = MllamaForConditionalGeneration.from_pretrained(
    model_file,
    torch_dtype=torch.bfloat16,
    device_map={"": device},
)
# processor = AutoProcessor.from_pretrained(model_file)



# 计算总花销
def calculate_total_expenses(user_input):
    input_content = "帮我计算：%s的金钱总额，将结果以“{\"total\":123.12}”格式返回，其中“123.12”为示例值，代表实际计算结果的数字。我不需要知道如何计算，只需要告诉我结果值即可。示例1：“吃饭8元的金钱总额，结果为{\"total\":8.00}”；示例2：“早餐10元，打车17元，买烟20元的金钱总额，结果为{\"total\":47.00}”；示例3：“早饭花了12元，买衣服花了120元，烟20元的金钱总额，结果为{\"total\":152.00}”；示例4:“早餐15元的金钱总额，结果为{\"total\":15.00}”。" % user_input
    logging.info(f"\ninput_content: {input_content}\n")
    # messages = [
    #     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    #     {"role": "user", "content": input_content},
    #     {"role": "assistant", "content": "你是Qwen的助手，你需要帮我计算金钱总额，注意数字的计算，不要算错。"}
    # ]
    messages = [
        {"role": "user", "content": [
            {"type": "json"},
            {"type": "text", "text": input_content}
        ]}
    ]


    # 准备对话内容
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.info(f"\ntext: {text}\n")
    inputs = tokenizer([text], return_tensors="pt").to(device)
    # 生成模型回复
    with torch.no_grad():
        # output = model.generate(**inputs, max_new_tokens=256) # max_new_tokens=512
        generated_ids = model.generate(**inputs, max_new_tokens=384)
    logging.info(f"\ngenerated_ids: {generated_ids}\n")

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    logging.info(f"\nlist of generated_ids: {generated_ids}\n")

    res_list = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    logging.info(f"\nres_list: {res_list}\n")
    response = res_list[0]
    logging.info(f"\nresponse: {response}\n")

    pattern = r'\{(.*?):(.*?)\}'
    # pattern = r'```json\s*(\{.*?\})\s*```'
    match = re.search(pattern, response)
    logging.info(f"\nmatch: {match}\n")
    json_text = match.group(0)
    logging.info(f"\njson_text: {json_text} \n")  # 输出 {"total":383.00"}
    json_data = json.loads(json_text)

    total_expenses = float(json_data["total"])
    logging.info(f"\ntotal_expenses: {total_expenses}\n")
    return total_expenses

# 将花销总额记录到Apple Calendar
def add_expense_to_apple_calendar(total_expenses, date):
    # 连接到iCloud Calendar
    client = DAVClient(url='https://caldav.icloud.com', username='orangejunehunt@163.com',
                       password='wgar-ddpi-riod-mvwf')
    principal = client.principal()

    # 创建事件
    calendar = Calendar()
    event = Event()
    event.add('uid', str(date.timestamp()) + '@icloud.com')
    event.add('dtstamp', date)
    event.add('dtstart', date)
    event.add('dtend', date + datetime.timedelta(hours=1))
    event.add('summary', f'今日总花费：{total_expenses}元')
    event.add('description', 'AI助手自动记录的花费')

    calendar.add_component(event)

    # 保存事件到日历
    principal.calendar().save_event(calendar.to_ical())


# Flask 路由
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        logging.info(f"\nuser_input: {user_input}\n")

        total_expenses = calculate_total_expenses(user_input)
        current_date = datetime.datetime.now()
        add_expense_to_apple_calendar(total_expenses, current_date)
        return render_template('result.html', total_expenses=total_expenses, calendar_message="已添加到Apple Calendar")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8087, use_reloader=False)
