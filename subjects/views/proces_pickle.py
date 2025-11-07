from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from data_analysis.utils import WESADProcessor 
from subjects.models import Subject
from django.conf import settings

class ImportDataAPIView(APIView):
    """
    API endpoint to trigger feature extraction for a specific subject ID.
    Accepts POST requests only.
    URL: /api/process/<subject_code>/
    """
    def post(self, request, subject_code, format=None):
        # 1. Validate Subject existence
        try:
            get_object_or_404(Subject, code=subject_code) 
        except Exception:
            return Response(
                {'status': 'error', 'message': f'Subject {subject_code} not found in database.'}, 
                status=status.HTTP_404_NOT_FOUND
            )

        # 2. Instantiate and process
        processor = WESADProcessor(subject_code, settings.DATA_DIR)
        success, message = processor.process()
        
        # 3. Return response
        if success:
            return Response(
                {'status': 'success', 'message': message}, 
                status=status.HTTP_200_OK
            )
        else:
            return Response(
                {'status': 'error', 'message': message}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )