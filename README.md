# Домашнее задание №1. Разработка Retrieval-Based чат-бота

## Задача:
Разработать чат-бот, используя подход retrieval-based. Бот должен вести диалог как определенный персонаж сериала, имитируя стиль и манеру конкретного персонажа сериала.

## Как запустить приложение в Docker:
1) Обучить модели, запустив Training.ipynb.
\
Чтобы не ждать, можно скачать обученные веса [отсюда](https://disk.yandex.ru/d/YeS6-71kWMeF3w) (git не даёт сохранять в репозиторий большие файлы). Разархивированную папку weights нужно разместить по пути app\static\weights

2) В корневой директории выполнить 
```bash
docker-compose up --build -d
```
3) Когда контейнер соберётся, некоторое время попрогружаются модели, и потом приложение с чат-ботом будет доступно на http://localhost:5000/

![Example_screen](app/static/images/example_dialog.png)

## Ход решения:
1) Взяла датасет с диалогами из Симпсонов на Kaggle:
https://www.kaggle.com/datasets/pierremegret/dialogue-lines-of-the-simpsons?resource=download. Сохранила его в директорию data.
2) Провела небольшой EDA (EDA.ipynb): посчитала кол-во реплик основных персонажей, построила облако слов для Гомера и вывела топ самых популярных фраз Гомера.
3) Experiments.ipynb - обучила нужные модели по примеру того, что было на семинарах.
4) Вынесла основные классы в отдельные файлы для возможности использовать их как в ноутбуках, так и в приложении:
    * app/biencoder/sentence_bert.py
    * app/dataset/homer_dataset.py
    * app/faiss/get_vocab.py
    * app/reranker/reranker.py
5) Добавила шаги по очистке данных при подготовке датасета (app.dataset.homer_dataset.prepare_data) и базы со словарём Гомера (app.faiss.get_vocab).
6) Обучила модели (Training.ipynb)
7) Потестировала, как генерируются ответы в другом ноутбуке (Testing.ipynb)
8) Оформила чатбот в виде приложения на Flask. Пример, как написать простой чат-бот в виде Flask app взяла отсюда: https://github.com/sahil-rajput/Candice-YourPersonalChatBot
