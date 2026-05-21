import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from dateutil.relativedelta import relativedelta
#本脚本专用于绘制甘特图
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题
tasks=[
    {"Task":"文献调研","Start":"2024-09","End":"2025-09"},
    {"Task":"RCWA算法实现","Start":"2025-09","End":"2026-01"},
    {"Task":"验证与调整算法","Start":"2026-01","End":"2026-06"},
    {"Task":"实验验证","Start":"2026-06","End":"2026-09"},
    {"Task":"论文撰写","Start":"2026-09","End":"2027-02"}
]

# for task in tasks:
#     start=datetime.strptime(task["Start"],"%Y-%m")
#     end=datetime.strptime(task["End"],"%Y-%m")+relativedelta(months=1)
#     task["start-date"]=start
#     task["end-date"]=end
#     task["duration"]=(end-start).days

fig,ax=plt.subplots(figsize=(10,4))
for i,task in enumerate(tasks):
    start=datetime.strptime(task["Start"],"%Y-%m")
    end=datetime.strptime(task["End"],"%Y-%m")
    ax.barh(task["Task"],(end-start).days,left=start,height=0.5,color="skyblue")
    # start_str=task["Start"]
    # end_str=task["End"]
    # x_center=task["start-date"]+(task["end-date"]-task["start-date"])/2
    # ax.text(
    #     x=task["start-date"],
    #     y=i,
    #     s=f"Start:{start_str}",
    #     va='center',
    #     ha='left',
    #     color='black',
    #     fontsize=9
    # )
    # ax.text(
    #     x=task["end-date"],
    #     y=i,
    #     s=f"End:{end_str}",
    #     va='center',
    #     ha='right',
    #     color='black',
    #     fontsize=9
    # )
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1,7)))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y-%m"))
plt.xlabel("时间轴")
plt.title("甘特图")
plt.tight_layout()
plt.show()