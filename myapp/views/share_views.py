from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect, get_object_or_404
from django.core.paginator import Paginator
from django.utils import timezone

from ..models import PhotoZoneData, ShareEmotion


# 각 detail 버튼을 클릭시, 각 detail 정보에 대해서 PhotoZoneData를 Question Form Field를 통해서 저장.

def share_index(request, video):
    if request.method == 'POST':
        subject = request.POST['subject']
        content = request.POST['content']
        data = PhotoZoneData.objects.filter(video__exact=video).first()
        share_emotion = ShareEmotion.objects.create(author=request.user, subject=subject, content=content, create_date=
                                        timezone.now(), share_emotion=data)
        return HttpResponseRedirect("/myapp/share/record")

    return render(request, 'myapp/share.html')


@login_required(login_url='common:login')
def share_record(request):
    page = request.GET.get('page', '1')  # 페이지

    share_emotion_list = ShareEmotion.objects.all()
    share_emotion_list = share_emotion_list.extra(order_by=['-create_date'])

    # 페이징처리
    paginator = Paginator(share_emotion_list, 10)  # 페이지당 10개씩 보여주기
    page_obj = paginator.get_page(page)

    context = {'share_emotion_list': page_obj, 'page': page}  # <------ so 추가
    return render(request, 'myapp/share_record.html', context)


def share_detail(request, share_emotion_id):
    share_emotion = get_object_or_404(ShareEmotion, pk=share_emotion_id)
    percentage = [0, 0, 0, 0, 0, 0, 0]
    # 비디오, 날짜, 감정레이블 데이터 필드 불러와서 다른 db에 저장.
    text_emotion = share_emotion.share_emotion.emotion_labeled_data
    for i in range(30):
        index = text_emotion[i]
        percentage[int(index)] += 1
    tmp_list = []
    for i in range(7):
        tmp_list.append((percentage[i] / 30) * 100)
    print(tmp_list)  # test

    context = {'share_emotion': share_emotion, 'share_emotion_percentage': tmp_list, "user": request.user}
    return render(request, 'myapp/share_detail.html', context)
