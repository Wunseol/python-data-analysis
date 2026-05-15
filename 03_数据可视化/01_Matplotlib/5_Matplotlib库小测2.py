import matplotlib.pyplot as plt

plt.plot([5,1,4,5,2])
plt.ylabel("Grade")
plt.savefig("test",dpi=600)
plt.show()

# plt.savefig()将输出图形存储为文件，默认PNG格式，可以通过dpi修改输出质量