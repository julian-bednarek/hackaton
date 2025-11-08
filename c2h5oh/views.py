from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from celery.result import AsyncResult
from .utils import process_pickle_data
from openai import OpenAI
from django.conf import settings
from io import BytesIO
from django.http import FileResponse
import json


class C2H5OHAppView(APIView):

    http_method_names = ["post", "options"]

    def _add_cors_headers(self, response):
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    def _validate_file(self, file_obj):
        if not file_obj:
            raise ValueError("No file provided.")
        if not file_obj.name.endswith(".pkl"):
            raise ValueError("Invalid file type. Only .pkl files are allowed.")

    def post(self, request):
        file_obj = request.FILES.get("file")
        try:
            self._validate_file(file_obj)
            audio_segment = process_pickle_data(file_obj)  # Returns AudioSegment
            wav_buffer = BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            response = FileResponse(
                wav_buffer,
                as_attachment=True,
                filename="processed_audio.wav",
                content_type="audio/wav",
            )
            return self._add_cors_headers(response)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response(
                {"error": "An unexpected error occurred: " + str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
