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
from config import model_file
from caldav import DAVClient
from icalendar import Calendar, Event

# Flask 初始化
app = Flask(__name__)

# 强制模型仅在GPU上运行
if torch.cuda.is_available():
    device = "cuda"  # 指定仅使用CUDA设备
else:
    raise RuntimeError("No GPU available. Please ensure a GPU is available for model execution.")


# 加载模型
tokenizer = AutoTokenizer.from_pretrained(model_file)
model = AutoModelForCausalLM.from_pretrained(model_file, torch_dtype=torch.float16, device_map={"": device}) # .to("cuda")


# 计算总花销
def calculate_total_expenses(user_input):
    input_content = "帮我计算金钱总额：%s，将结果以“{'total':xxx.xx}”格式返回，其中“xxx.xx”为计算结果的数字。" % user_input
    print(f"\ninput_content: {input_content}\n")
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": input_content}
    ]
    # 准备对话内容
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # 生成模型回复
    with torch.no_grad():
        # output = model.generate(**inputs, max_new_tokens=256) # max_new_tokens=512
        generated_ids = model.generate(**inputs, max_new_tokens=384)

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    res_list = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    response = res_list[0]
    print(f"\nresponse: {response}\n")

    pattern = r'\{(.*?):(.*?)\}'
    # pattern = r'```json\s*(\{.*?\})\s*```'
    match = re.search(pattern, response)
    print(f"\nmatch: {match}\n")
    json_text = match.group(0)
    print(f"\njson_text: {json_text} \n")  # 输出 {"total":383.00"}
    json_data = json.loads(json_text)

    total_expenses = float(json_data["total"])
    print(f"\ntotal_expenses: {total_expenses}\n")
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
        print(f"\nuser_input: {user_input}\n")

        total_expenses = calculate_total_expenses(user_input)
        current_date = datetime.datetime.now()
        add_expense_to_apple_calendar(total_expenses, current_date)
        return render_template('result.html', total_expenses=total_expenses, calendar_message="已添加到Apple Calendar")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8087, use_reloader=False)
