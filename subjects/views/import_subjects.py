from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from data_analysis.utils import ReadmeImporter

from django.conf import settings

class ImportReadmeDataAPIView(APIView):
    def post(self, request, subject_code, format=None):
        
        importer = ReadmeImporter(subject_code, data_directory=settings.DATA_DIR)
        success, message = importer.import_and_update()

        if success:
            return Response(
                {'status': 'success', 'message': message}, 
                status=status.HTTP_200_OK
            )
        else:
            if "not found" in message:
                 return Response(
                    {'status': 'error', 'message': message}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            return Response(
                {'status': 'error', 'message': message}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )