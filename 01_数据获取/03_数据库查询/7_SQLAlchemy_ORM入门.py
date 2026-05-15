# 数据源: SQLite内存数据库 (无需外部数据库配置)
# [注意] 本系列使用 SQLAlchemy 2.0 语法风格，与 1.x 有较大差异

"""
SQLAlchemy ORM 入门 (2.0 语法风格)
- create_engine 创建引擎
- DeclarativeBase 声明基类
- Column 列类型定义
- Session 会话管理
- 添加对象 (add/add_all)
- 查询: select(), filter, all/first/scalar
"""

from sqlalchemy import create_engine, String, Integer, Float, ForeignKey, select, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, relationship


class Base(DeclarativeBase):
    pass


class Department(Base):
    __tablename__ = "departments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    location: Mapped[str | None] = mapped_column(String(100))

    employees: Mapped[list["Employee"]] = relationship(back_populates="department")

    def __repr__(self):
        return f"Department(id={self.id}, name='{self.name}', location='{self.location}')"


class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    salary: Mapped[float] = mapped_column(Float, nullable=False)
    department_id: Mapped[int | None] = mapped_column(ForeignKey("departments.id"))

    department: Mapped[Department | None] = relationship(back_populates="employees")

    def __repr__(self):
        return f"Employee(id={self.id}, name='{self.name}', salary={self.salary}, dept_id={self.department_id})"


def create_engine_demo():
    print("=" * 60)
    print("1. create_engine 创建引擎")
    print("=" * 60)

    # SQLite内存数据库引擎
    # echo=True 会打印SQL语句，便于调试
    engine = create_engine("sqlite:///:memory:", echo=False)
    print(f"引擎对象: {engine}")
    print(f"数据库URL: {engine.url}")

    # 创建所有表 (根据Base的元数据)
    Base.metadata.create_all(engine)
    print("已创建所有表: departments, employees")

    return engine


def add_objects(engine):
    print("=" * 60)
    print("2. 添加对象: Session.add() / Session.add_all()")
    print("=" * 60)

    # 使用 Session 管理数据库会话
    # SQLAlchemy 2.0 推荐使用 context manager
    with Session(engine) as session:
        # 创建部门对象
        tech_dept = Department(name="技术部", location="北京")
        market_dept = Department(name="市场部", location="上海")
        finance_dept = Department(name="财务部", location="广州")

        # add_all 批量添加
        session.add_all([tech_dept, market_dept, finance_dept])

        # 创建员工对象
        emp1 = Employee(name="张三", salary=25000.0, department_id=1)
        emp2 = Employee(name="李四", salary=20000.0, department_id=1)
        emp3 = Employee(name="王五", salary=18000.0, department_id=2)
        emp4 = Employee(name="赵六", salary=15000.0, department_id=2)
        emp5 = Employee(name="孙七", salary=16000.0, department_id=3)

        session.add_all([emp1, emp2, emp3, emp4, emp5])

        # 提交事务
        session.commit()
        print("已添加3个部门和5个员工")

        # 提交后对象会获得自动生成的id
        print(f"  {tech_dept}")
        print(f"  {emp1}")


def query_select_all(engine):
    print("=" * 60)
    print("3. 查询: select() 获取所有记录")
    print("=" * 60)

    with Session(engine) as session:
        # SQLAlchemy 2.0 风格: 使用 select() 构造查询
        stmt = select(Department)
        result = session.execute(stmt)

        # all(): 获取所有结果
        departments = result.all()
        print("所有部门:")
        for dept in departments:
            print(f"  {dept}")

        # 也可以直接用 scalars() 获取标量对象列表
        stmt = select(Employee)
        employees = session.scalars(stmt).all()
        print("\n所有员工:")
        for emp in employees:
            print(f"  {emp}")


def query_filter(engine):
    print("=" * 60)
    print("4. 查询: filter 条件筛选")
    print("=" * 60)

    with Session(engine) as session:
        # 等值条件
        stmt = select(Employee).where(Employee.department_id == 1)
        tech_employees = session.scalars(stmt).all()
        print("技术部员工 (department_id == 1):")
        for emp in tech_employees:
            print(f"  {emp}")

        # 比较条件
        stmt = select(Employee).where(Employee.salary > 18000)
        high_salary = session.scalars(stmt).all()
        print("\n薪资 > 18000:")
        for emp in high_salary:
            print(f"  {emp}")

        # 多条件组合 (AND)
        stmt = select(Employee).where(
            Employee.department_id == 1,
            Employee.salary > 22000,
        )
        result = session.scalars(stmt).all()
        print("\n技术部且薪资>22000:")
        for emp in result:
            print(f"  {emp}")

        # 使用 or_
        from sqlalchemy import or_
        stmt = select(Employee).where(
            or_(Employee.salary > 20000, Employee.salary < 16000)
        )
        result = session.scalars(stmt).all()
        print("\n薪资>20000 或 薪资<16000:")
        for emp in result:
            print(f"  {emp}")

        # LIKE 模糊查询
        stmt = select(Employee).where(Employee.name.like("张%"))
        result = session.scalars(stmt).all()
        print("\n姓'张'的员工:")
        for emp in result:
            print(f"  {emp}")


def query_first_scalar(engine):
    print("=" * 60)
    print("5. 查询: first() / scalar() / one()")
    print("=" * 60)

    with Session(engine) as session:
        # first(): 获取第一条结果，无结果返回None
        stmt = select(Employee).where(Employee.salary > 20000)
        first_emp = session.scalars(stmt).first()
        print(f"first() - 薪资>20000的第一人: {first_emp}")

        # scalar(): 获取单个值，常用于聚合查询
        stmt = select(func.count()).select_from(Employee)
        count = session.scalar(stmt)
        print(f"scalar() - 员工总数: {count}")

        stmt = select(func.avg(Employee.salary))
        avg_salary = session.scalar(stmt)
        print(f"scalar() - 平均薪资: {avg_salary:.2f}")

        # 获取单个对象 (通过主键)
        stmt = select(Employee).where(Employee.id == 1)
        emp = session.scalars(stmt).one()
        print(f"one() - id=1的员工: {emp}")


def query_order_limit(engine):
    print("=" * 60)
    print("6. 查询: order_by 排序 与 limit 限制")
    print("=" * 60)

    with Session(engine) as session:
        # 升序
        stmt = select(Employee).order_by(Employee.salary.asc())
        result = session.scalars(stmt).all()
        print("薪资升序:")
        for emp in result:
            print(f"  {emp.name}: {emp.salary}")

        # 降序 + limit
        stmt = select(Employee).order_by(Employee.salary.desc()).limit(3)
        result = session.scalars(stmt).all()
        print("\n薪资最高的3人:")
        for emp in result:
            print(f"  {emp.name}: {emp.salary}")

        # offset 分页
        stmt = select(Employee).order_by(Employee.id).offset(2).limit(2)
        result = session.scalars(stmt).all()
        print("\n第2页 (每页2条):")
        for emp in result:
            print(f"  {emp.name}: {emp.salary}")


def query_join(engine):
    print("=" * 60)
    print("7. 查询: join 关联查询")
    print("=" * 60)

    with Session(engine) as session:
        # 显式 join
        stmt = (
            select(Employee.name, Department.name, Employee.salary)
            .join(Department, Employee.department_id == Department.id)
            .order_by(Employee.salary.desc())
        )
        result = session.execute(stmt).all()
        print("员工-部门 join 查询:")
        print(f"  {'员工':<6} {'部门':<8} {'薪资':<10}")
        for name, dept_name, salary in result:
            print(f"  {name:<6} {dept_name:<8} {salary:<10}")

        # 使用 relationship 自动加载
        stmt = select(Employee)
        for emp in session.scalars(stmt):
            dept_name = emp.department.name if emp.department else "(无部门)"
            print(f"  {emp.name} -> {dept_name}")


def query_aggregate(engine):
    print("=" * 60)
    print("8. 查询: 聚合函数")
    print("=" * 60)

    with Session(engine) as session:
        # GROUP BY 聚合
        stmt = (
            select(
                Department.name,
                func.count(Employee.id).label("employee_count"),
                func.avg(Employee.salary).label("avg_salary"),
                func.sum(Employee.salary).label("total_salary"),
            )
            .join(Department, Employee.department_id == Department.id)
            .group_by(Department.name)
        )
        result = session.execute(stmt).all()
        print("各部门薪资统计:")
        print(f"  {'部门':<8} {'人数':<6} {'平均薪资':<12} {'总薪资':<12}")
        for dept_name, count, avg_s, total_s in result:
            print(f"  {dept_name:<8} {count:<6} {avg_s:<12.0f} {total_s:<12.0f}")


def update_delete(engine):
    print("=" * 60)
    print("9. 更新与删除对象")
    print("=" * 60)

    with Session(engine) as session:
        # 更新: 先查询，再修改属性，最后commit
        stmt = select(Employee).where(Employee.name == "张三")
        emp = session.scalars(stmt).one()
        print(f"更新前: {emp}")
        emp.salary = 28000.0
        session.commit()
        print(f"更新后: {emp}")

        # 批量更新 (使用 update)
        from sqlalchemy import update
        stmt = update(Employee).where(Employee.department_id == 2).values(salary=Employee.salary * 1.1)
        session.execute(stmt)
        session.commit()
        print("\n市场部员工加薪10%后:")
        stmt = select(Employee).where(Employee.department_id == 2)
        for emp in session.scalars(stmt):
            print(f"  {emp.name}: {emp.salary}")

        # 删除
        stmt = select(Employee).where(Employee.name == "赵六")
        emp_to_delete = session.scalars(stmt).first()
        if emp_to_delete:
            session.delete(emp_to_delete)
            session.commit()
            print(f"\n已删除: {emp_to_delete}")

        # 验证
        stmt = select(func.count()).select_from(Employee)
        count = session.scalar(stmt)
        print(f"剩余员工数: {count}")


if __name__ == "__main__":
    engine = create_engine_demo()
    add_objects(engine)
    query_select_all(engine)
    query_filter(engine)
    query_first_scalar(engine)
    query_order_limit(engine)
    query_join(engine)
    query_aggregate(engine)
    update_delete(engine)
    print("\n所有SQLAlchemy ORM入门演示完成！")
