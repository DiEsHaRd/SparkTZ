from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit, when


def get_products_with_categories(products_df: DataFrame,
                                 categories_df: DataFrame,
                                 product_category_df: DataFrame) -> DataFrame:
    """
    Возвращает датафрейм со всеми парами «Имя продукта – Имя категории»
    и продуктами без категорий.

    Args:
        products_df: Датафрейм продуктов с колонками ['product_id', 'product_name']
        categories_df: Датафрейм категорий с колонками ['category_id', 'category_name']
        product_category_df: Датафрейм связей с колонками ['product_id', 'category_id']

    Returns:
        Датафрейм с колонками ['product_name', 'category_name']
    """
    # Проверка наличия необходимых колонок
    required_product_cols = ['product_id', 'product_name']
    required_category_cols = ['category_id', 'category_name']
    required_link_cols = ['product_id', 'category_id']

    for col_name in required_product_cols:
        if col_name not in products_df.columns:
            raise ValueError(f"В products_df отсутствует колонка: {col_name}")

    for col_name in required_category_cols:
        if col_name not in categories_df.columns:
            raise ValueError(f"В categories_df отсутствует колонка: {col_name}")

    for col_name in required_link_cols:
        if col_name not in product_category_df.columns:
            raise ValueError(f"В product_category_df отсутствует колонка: {col_name}")

    # Соединяем продукты с категориями через таблицу связей
    products_with_categories = (products_df
                                .join(product_category_df, 'product_id', 'left')
                                .join(categories_df, 'category_id', 'left')
                                .select('product_name', 'category_name')
                                )

    # Добавляем продукты без категорий (category_name = NULL)
    # и заменяем NULL на указание отсутствия категории
    result_df = products_with_categories.withColumn(
        'category_name',
        when(col('category_name').isNull(), lit('No category'))
        .otherwise(col('category_name'))
    )

    return result_df.orderBy('product_name', 'category_name')


if __name__ == "__main__":
    # Пример использования
    spark = SparkSession.builder \
        .appName("ProductsCategoriesExample") \
        .master("local[2]") \
        .getOrCreate()

    try:
        # Создание тестовых данных
        products_df = spark.createDataFrame([
            (1, "Laptop"),
            (2, "Smartphone"),
            (3, "Book"),
            (4, "Chair")  # Продукт без категории
        ], ["product_id", "product_name"])

        categories_df = spark.createDataFrame([
            (1, "Electronics"),
            (2, "Computers"),
            (3, "Books")
        ], ["category_id", "category_name"])

        links_df = spark.createDataFrame([
            (1, 1),  # Laptop - Electronics
            (1, 2),  # Laptop - Computers
            (2, 1),  # Smartphone - Electronics
            (3, 3)  # Book - Books
            # Chair не имеет категорий
        ], ["product_id", "category_id"])

        # Выполнение метода
        print("Входные данные:")
        print("Продукты:")
        products_df.show()
        print("Категории:")
        categories_df.show()
        print("Связи:")
        links_df.show()

        print("Результат:")
        result = get_products_with_categories(products_df, categories_df, links_df)
        result.show()

    finally:
        spark.stop()