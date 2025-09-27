import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from products_categories import get_products_with_categories


class TestProductsWithCategories:

    @pytest.fixture(scope="class")
    def spark(self):
        # Создание Spark сессии для тестов
        spark = SparkSession.builder \
            .appName("test_products_categories") \
            .master("local[2]") \
            .config("spark.sql.adaptive.enabled", "false") \
            .getOrCreate()

        yield spark
        spark.stop()

    @pytest.fixture
    def sample_data(self, spark):
        # Создание тестовых данных
        products_data = [
            (1, "Laptop"),
            (2, "Smartphone"),
            (3, "Book"),
            (4, "Chair"),
            (5, "Tablet")
        ]

        categories_data = [
            (1, "Electronics"),
            (2, "Computers"),
            (3, "Furniture"),
            (4, "Books")
        ]

        links_data = [
            (1, 1),  # Laptop - Electronics
            (1, 2),  # Laptop - Computers
            (2, 1),  # Smartphone - Electronics
            (3, 4),  # Book - Books
            (5, 1),  # Tablet - Electronics
            # Chair не имеет категорий
        ]

        products_df = spark.createDataFrame(products_data, ["product_id", "product_name"])
        categories_df = spark.createDataFrame(categories_data, ["category_id", "category_name"])
        links_df = spark.createDataFrame(links_data, ["product_id", "category_id"])

        return products_df, categories_df, links_df

    def test_basic_functionality(self, spark, sample_data):
        # Тест базовой функциональности
        products_df, categories_df, links_df = sample_data

        result_df = get_products_with_categories(products_df, categories_df, links_df)
        result = result_df.collect()

        # Проверяем количество строк
        assert result_df.count() == 6

        # Проверяем ожидаемые пары
        expected_pairs = [
            ("Book", "Books"),
            ("Chair", "No category"),
            ("Laptop", "Computers"),
            ("Laptop", "Electronics"),
            ("Smartphone", "Electronics"),
            ("Tablet", "Electronics")
        ]

        actual_pairs = [(row['product_name'], row['category_name']) for row in result]

        # Сортируем для сравнения
        expected_pairs.sort()
        actual_pairs.sort()

        assert actual_pairs == expected_pairs

    def test_products_without_categories(self, spark):
        # Тест когда у продуктов нет категорий
        products_data = [(1, "Product1"), (2, "Product2")]
        categories_data = [(1, "Category1")]
        links_data = []  # Нет связей

        products_df = spark.createDataFrame(products_data, ["product_id", "product_name"])
        categories_df = spark.createDataFrame(categories_data, ["category_id", "category_name"])
        links_df = spark.createDataFrame(links_data, ["product_id", "category_id"])

        result_df = get_products_with_categories(products_df, categories_df, links_df)
        result = result_df.collect()

        assert result_df.count() == 2
        assert all(row['category_name'] == 'No category' for row in result)

    def test_categories_without_products(self, spark):
        # Тест когда есть категории без продуктов
        products_data = [(1, "Product1")]
        categories_data = [(1, "Category1"), (2, "Category2")]
        links_data = [(1, 1)]

        products_df = spark.createDataFrame(products_data, ["product_id", "product_name"])
        categories_df = spark.createDataFrame(categories_data, ["category_id", "category_name"])
        links_df = spark.createDataFrame(links_data, ["product_id", "category_id"])

        result_df = get_products_with_categories(products_df, categories_df, links_df)

        assert result_df.count() == 1
        assert result_df.collect()[0]['category_name'] == 'Category1'

    def test_empty_dataframes(self, spark):
        # Тест с пустыми датафреймами
        empty_schema = StructType([
            StructField("product_id", IntegerType(), True),
            StructField("product_name", StringType(), True)
        ])

        products_df = spark.createDataFrame([], empty_schema)
        categories_df = spark.createDataFrame([], empty_schema)
        links_df = spark.createDataFrame([], empty_schema)

        result_df = get_products_with_categories(products_df, categories_df, links_df)
        assert result_df.count() == 0

    def test_missing_columns(self, spark):
        # Тест обработки отсутствующих колонок
        products_data = [(1, "Product1")]
        products_df = spark.createDataFrame(products_data, ["id", "name"])

        categories_df = spark.createDataFrame([(1, "Cat1")], ["category_id", "category_name"])
        links_df = spark.createDataFrame([(1, 1)], ["product_id", "category_id"])

        with pytest.raises(ValueError, match="В products_df отсутствует колонка: product_id"):
            get_products_with_categories(products_df, categories_df, links_df)

    def test_duplicate_relationships(self, spark):
        # Тест дублирующихся связей
        products_data = [(1, "Product1")]
        categories_data = [(1, "Category1")]
        links_data = [(1, 1), (1, 1)]

        products_df = spark.createDataFrame(products_data, ["product_id", "product_name"])
        categories_df = spark.createDataFrame(categories_data, ["category_id", "category_name"])
        links_df = spark.createDataFrame(links_data, ["product_id", "category_id"])

        result_df = get_products_with_categories(products_df, categories_df, links_df)
        assert result_df.count() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])