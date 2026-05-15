# 依赖库最低版本要求: pandas>=2.0
# 数据来源: 本文件使用 pandas 构造的模拟数据集，无需外部数据文件
# 注意: 本文件中的邮箱和密码均为占位符，请替换为实际凭据后使用

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from pathlib import Path
from datetime import datetime

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

print("=" * 60)
print("6. 邮件自动发送报告")
print("=" * 60)

# --------------------------------------------------
# 一、邮件配置 (占位符，请替换为实际信息)
# --------------------------------------------------

SMTP_SERVER = "smtp.example.com"
SMTP_PORT = 465
SENDER_EMAIL = "sender@example.com"
SENDER_PASSWORD = "your_app_password_here"

RECEIVER_EMAILS = [
    "manager@example.com",
    "team_lead@example.com",
]

print("[配置] SMTP服务器:", SMTP_SERVER)
print("[配置] 发件人:", SENDER_EMAIL)
print("[配置] 收件人:", RECEIVER_EMAILS)
print("[警告] 请将占位符替换为实际邮箱和密码后再运行发送部分")

# --------------------------------------------------
# 二、创建模拟附件文件
# --------------------------------------------------

import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "姓名": [f"员工{i:02d}" for i in range(1, 21)],
    "部门": np.random.choice(["技术部", "市场部", "财务部", "人事部"], 20),
    "月薪": np.random.randint(8000, 25000, 20),
    "绩效评分": np.round(np.random.uniform(60, 100, 20), 1),
})

csv_attachment = output_dir / "周报数据.csv"
df.to_csv(csv_attachment, index=False, encoding="utf-8-sig")
print(f"\n[附件] CSV文件已准备: {csv_attachment}")

excel_attachment = output_dir / "周报数据.xlsx"
df.to_excel(excel_attachment, index=False, engine="openpyxl")
print(f"[附件] Excel文件已准备: {excel_attachment}")

# --------------------------------------------------
# 三、构建 HTML 邮件正文
# --------------------------------------------------

report_date = datetime.now().strftime("%Y年%m月%d日")
dept_summary = df.groupby("部门").agg(
    人数=("姓名", "count"),
    平均月薪=("月薪", "mean"),
    平均绩效=("绩效评分", "mean"),
).round(2)

table_rows = ""
for dept, row in dept_summary.iterrows():
    table_rows += f"""
    <tr>
        <td style="padding:8px;border:1px solid #ddd;text-align:center;">{dept}</td>
        <td style="padding:8px;border:1px solid #ddd;text-align:center;">{int(row['人数'])}</td>
        <td style="padding:8px;border:1px solid #ddd;text-align:right;">{row['平均月薪']:,.2f} 元</td>
        <td style="padding:8px;border:1px solid #ddd;text-align:center;">{row['平均绩效']:.2f}</td>
    </tr>"""

html_body = f"""
<html>
<body style="font-family: 'Microsoft YaHei', Arial, sans-serif; color: #333; line-height: 1.6;">
    <div style="max-width: 700px; margin: 0 auto;">
        <h2 style="color: #2F5496; border-bottom: 2px solid #2F5496; padding-bottom: 10px;">
            员工数据周报
        </h2>
        <p>各位领导好，</p>
        <p>以下是 <strong>{report_date}</strong> 的员工数据周报，请查阅。</p>

        <h3 style="color: #2F5496;">数据概要</h3>
        <ul>
            <li>员工总数: <strong>{len(df)}</strong> 人</li>
            <li>平均月薪: <strong>{df['月薪'].mean():,.0f}</strong> 元</li>
            <li>平均绩效: <strong>{df['绩效评分'].mean():.1f}</strong> 分</li>
        </ul>

        <h3 style="color: #2F5496;">部门汇总</h3>
        <table style="border-collapse:collapse;width:100%;margin-bottom:20px;">
            <thead>
                <tr style="background-color:#2F5496;color:white;">
                    <th style="padding:10px;border:1px solid #ddd;">部门</th>
                    <th style="padding:10px;border:1px solid #ddd;">人数</th>
                    <th style="padding:10px;border:1px solid #ddd;">平均月薪</th>
                    <th style="padding:10px;border:1px solid #ddd;">平均绩效</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>

        <p style="color:#666;font-size:12px;">
            详细数据请见附件。本邮件由系统自动发送，如有问题请联系数据分析团队。
        </p>
    </div>
</body>
</html>
"""

print("[邮件] HTML正文已构建")

# --------------------------------------------------
# 四、构建纯文本备选正文
# --------------------------------------------------

text_body = f"""员工数据周报 - {report_date}

数据概要:
- 员工总数: {len(df)} 人
- 平均月薪: {df['月薪'].mean():,.0f} 元
- 平均绩效: {df['绩效评分'].mean():.1f} 分

部门汇总:
"""
for dept, row in dept_summary.iterrows():
    text_body += f"  {dept}: {int(row['人数'])}人, 平均月薪 {row['平均月薪']:,.0f} 元, 平均绩效 {row['平均绩效']:.1f}\n"

text_body += "\n详细数据请见附件。本邮件由系统自动发送。"

print("[邮件] 纯文本正文已构建")

# --------------------------------------------------
# 五、组装邮件 (MIMEMultipart)
# --------------------------------------------------

def build_email(sender, receivers, subject, text_content, html_content, attachments=None):
    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = ", ".join(receivers)
    msg["Subject"] = subject
    msg["Date"] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0800")

    msg.attach(MIMEText(text_content, "plain", "utf-8"))
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    if attachments:
        for file_path in attachments:
            file_path = Path(file_path)
            with open(file_path, "rb") as f:
                part = MIMEApplication(f.read(), Name=file_path.name)
            part["Content-Disposition"] = f'attachment; filename="{file_path.name}"'
            msg.attach(part)

    return msg

attachments = [csv_attachment, excel_attachment]
email_msg = build_email(
    sender=SENDER_EMAIL,
    receivers=RECEIVER_EMAILS,
    subject=f"员工数据周报 - {report_date}",
    text_content=text_body,
    html_content=html_body,
    attachments=attachments,
)

print(f"\n[邮件] 已组装完成")
print(f"  发件人: {email_msg['From']}")
print(f"  收件人: {email_msg['To']}")
print(f"  主题: {email_msg['Subject']}")
print(f"  附件数: {len(attachments)}")

# --------------------------------------------------
# 六、发送邮件 (SSL/TLS)
# --------------------------------------------------

def send_email_ssl(msg, smtp_server, smtp_port, sender, password):
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
        server.login(sender, password)
        server.send_message(msg)
    print("[发送] 邮件已通过 SSL 发送成功")

def send_email_tls(msg, smtp_server, smtp_port, sender, password):
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(sender, password)
        server.send_message(msg)
    print("[发送] 邮件已通过 TLS 发送成功")

DEMO_MODE = True

if DEMO_MODE:
    print("\n[演示模式] 跳过实际发送 (DEMO_MODE = True)")
    print("  如需实际发送，请:")
    print("  1. 将 DEMO_MODE 改为 False")
    print("  2. 替换 SMTP_SERVER, SENDER_EMAIL, SENDER_PASSWORD 为实际值")
    print("  3. 替换 RECEIVER_EMAILS 为实际收件人")
    print("  4. 选择发送方式: send_email_ssl() 或 send_email_tls()")
else:
    try:
        send_email_ssl(
            email_msg, SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD
        )
    except Exception as e:
        print(f"[错误] 发送失败: {e}")

# --------------------------------------------------
# 七、邮件内容预览
# --------------------------------------------------

print("\n--- 纯文本正文预览 ---")
print(text_body)

print("\n--- HTML正文预览 (前500字符) ---")
print(html_body[:500] + "...")

print("\n" + "=" * 60)
print("邮件自动发送报告 - 完成")
print("=" * 60)
