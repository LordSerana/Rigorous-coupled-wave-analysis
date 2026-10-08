import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta
#本脚本专用于绘制甘特图
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题
tasks=[
    {"Task":"RCWA对多层介质膜结构和\n含粗糙度结构的计算研究","Start":"2026-10","End":"2026-12"},
    {"Task":"RCWA代码重构","Start":"2026-12","End":"2027-02"},
    {"Task":"论文撰写","Start":"2027-02","End":"2027-05"}
]

fig,ax=plt.subplots(figsize=(12,4.5))
for i,task in enumerate(tasks):
    start=datetime.strptime(task["Start"],"%Y-%m")
    end=datetime.strptime(task["End"],"%Y-%m")
    end_bar=end+relativedelta(months=1)
    duration=(end_bar-start).days
    #绘制条形
    ax.barh(i,duration,left=start,height=0.55,color="skyblue",edgecolor="black",linewidth=1)
    #在开始处标注
    ax.text(start,i,f"{task['Start']}",va="center",ha="left",fontsize=9,color="black")
    #在结束处标注
    ax.text(end_bar,i,f"{task['End']}",va="center",ha="right",fontsize=9,color="black")
#设置y轴显示任务名称
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([t["Task"] for t in tasks])
ax.invert_yaxis()
#设置时间轴刻度
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
#自动扩展x轴范围
x_start=datetime(2026,9,20)
x_end=datetime(2027,6,10)
ax.set_xlim(x_start,x_end)
ax.grid(axis="x",linestyle="--",alpha=0.5)
ax.set_axisbelow(True)
ax.set_xlabel("时间",fontsize=12)
ax.set_title("研究工作甘特图",fontsize=15,pad=15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()