# Git测试文件
# 用于测试Git功能的简单图形输出代码

def draw_tree():
    """绘制一个简单的ASCII树"""
    print("    🌳")
    print("   /|\\")
    print("  / | \\")
    print(" /  |  \\")
    print("/___|___\\")

def draw_heart():
    """绘制一个简单的ASCII心形"""
    print("  ♥♥   ♥♥")
    print("♥    ♥    ♥")
    print("♥         ♥")
    print("  ♥     ♥")
    print("    ♥ ♥")
    print("     ♥")

def draw_git_logo():
    """绘制一个简单的Git标志"""
    print("  ________")
    print(" /        \\")
    print("|   Git    |")
    print(" \\________/")
    print("    |  |")
    print("    |  |")
    print("    |  |")

if __name__ == "__main__":
    print("Git测试 - 图形输出")
    print("=" * 20)

    print("\n树:")
    draw_tree()

    print("\n心形:")
    draw_heart()

    print("\nGit标志:")
    draw_git_logo()

    print("\n测试完成!")
