from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.core.paginator import Paginator
from ..models import PhotoZoneData


@login_required(login_url='common:login')
def record_index(request):
    page = request.GET.get('page', '1')  # 페이지

    record_data_list = PhotoZoneData.objects.filter(owner__exact=request.user)
    record_data_list = record_data_list.extra(order_by=['-create_date'])

    paginator = Paginator(record_data_list, 10)  # 페이지당 10개씩 보여주기
    page_obj = paginator.get_page(page)

    context = {'record_data_list': page_obj, 'page': page}
    return render(request, "myapp/record.html", context)


def statistics(request):
    # 여기서 감정 데이터 레이블을 날짜에 따라 받아서 db 에 저장. 동기화
    # 동기화를 진행하기 위해서 각 날짜에 따른 감정 레이블 데이터를 실제 감정 상태로 매팅 변환하여,
    # 날짜에 따른 각 실제 감정 상태 퍼센티지를 따로 다른 db에 저장.

    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    statistics_raw_data = PhotoZoneData.objects.filter(owner__exact=request.user)
    statistics_raw_data = statistics_raw_data.extra(order_by=['-create_date'])
    graph_data = {}

    for data in statistics_raw_data:
        percentage = [0, 0, 0, 0, 0, 0, 0]
        # 비디오, 날짜, 감정레이블 데이터 필드 불러와서 다른 db에 저장.
        text_emotion = data.emotion_labeled_data
        for i in range(30):
            index = text_emotion[i]
            percentage[int(index)] += 1
        tmp_list = []
        for i in range(7):
            tmp_list.append((percentage[i] / 30) * 100)
        # print(percentage)
        # print(tmp_list)
        graph_data[data.datetime] = tmp_list

    # 날짜에 따른 statistics를 표현하는 막대 그래프
    #  시간에 따라 순차적 정렬 후, 감정을
    # context에 막대그래프 데이터 형식으로 매핑 시켜서 template에 넘겨준다.
    graph_data_x = []
    graph_data_y = []
    for key in graph_data.keys():
        graph_data_x.append(key)
    for value in graph_data.values():
        graph_data_y.append(value)
    context = {"graph_data_x": graph_data_x, "graph_data_y": graph_data_y}
    return render(request, "myapp/statistic.html", context)


def video_cv_open(request, video):
    # 현재 페이지 날짜에 따른 감정 데이터 레이블 퍼센티지
    # 여기서 감정 데이터 레이블을 날짜(video이름 파싱을 통해 얻을 것.)에 따라 받아서 레이블 데이터를 template에 보내줌.

    statistics_raw_data = PhotoZoneData.objects.filter(owner__exact=request.user)
    tmp_list = []

    for data in statistics_raw_data:
        if data.video == video:
            percentage = [0, 0, 0, 0, 0, 0, 0]
            # 비디오, 날짜, 감정레이블 데이터 필드 불러와서 다른 db에 저장.
            text_emotion = data.emotion_labeled_data
            for i in range(30):
                index = text_emotion[i]
                percentage[int(index)] += 1
            # tmp_list = []
            for i in range(7):
                tmp_list.append((percentage[i] / 30) * 100)

    print(tmp_list)
    context = {'video': video, 'percentage': tmp_list}
    print(video)
    return render(request, "myapp/video.html", context)
