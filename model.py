# coding: utf-8
# @Author: WalkerZ
# @Time: 2024/10/16

from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
from dateutil import parser
import datetime
import re
from config import model_file
from caldav import DAVClient
from icalendar import Calendar, Event

# Flask 初始化
app = Flask(__name__)

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(model_file)
model = AutoModelForCausalLM.from_pretrained(model_file) # .to("cuda")

# 计算总花销
def calculate_total_expenses(user_input):
    # inputs = tokenizer(user_input, return_tensors="pt")  #.to("cuda")
    # outputs = model.generate(**inputs)
    #
    # output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # # 替换中文逗号为英文逗号
    # output_str = output_str.replace('，', ',')
    # total_expenses = eval(output_str)
    #
    # # total_expenses = eval(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # return total_expenses
    inputs = tokenizer(user_input, return_tensors="pt")  # .to("cuda")
    outputs = model.generate(**inputs)
    output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 替换中文逗号为英文逗号
    output_str = output_str.replace('，', ',')

    # 使用正则表达式提取金额
    amounts = re.findall(r'(\d+\.?\d*)\s*元', output_str)  # 提取所有“X元”的金额
    total_expenses = sum(float(amount) for amount in amounts)  # 计算总金额

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
        total_expenses = calculate_total_expenses(user_input)
        current_date = datetime.datetime.now()
        add_expense_to_apple_calendar(total_expenses, current_date)
        return render_template('result.html', total_expenses=total_expenses, calendar_message="已添加到Apple Calendar")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8087, use_reloader=False)
