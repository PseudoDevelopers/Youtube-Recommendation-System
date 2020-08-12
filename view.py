from django.http import HttpResponse
from django.shortcuts import render

from processing.recommendations import recommender

video_recommender = recommender()


def home(request):
    videos = {
        'videos': video_recommender.get_random_videos()
    }

    return render(request, 'home.html', videos)

def search(request):
    query = request.GET.get('q')
    data = {
        'videos': video_recommender.search_videos(query),
        'query': query
    }

    return render(request, 'search.html', data)


def watch(request):
    requested_video_id = request.GET.get('v')
    videos = video_recommender.get_recommended_videos(requested_video_id)

    return render(request, 'watch.html', videos)
