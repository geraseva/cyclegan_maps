# cyclegan_maps
Using cycleGAN to transform satellite views into DnD maps

С помошью CycleGAN я пыталась превратить спутниковые снимки в карты для DnD. Ссылка на проект на github: https://github.com/geraseva/cyclegan_maps/.

В папке scripts лежат коды для обучения сети (cyclegan.py), для трансформации одной картинки (apply.py), загрузчики датасетов с картинками (data_loader.py) и архитектуры сетей (networks.py).

В папке datasets лежат датасеты для обучения. В папке google_maps лежат спутниковые снимки, этот датасет я брала отсюда: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/maps.zip. В папке city_maps лежат карты городов для DnD. Их я скачивала отсюда https://www.reddit.com/r/dndmaps/ с помошью этой программы https://github.com/sammax/reddit-image-downloader. В папке city_maps_new лежат карты городов, порезанные до нужного масштаба (вручную) и приведенные к размеру 600 на 600.

В папке weights лежат веса для модели. 

Команды для запуска модели в ноутбуке.
