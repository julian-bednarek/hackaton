from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from celery.result import AsyncResult
from .utils import process_pickle_data

class C2H5OHAppView(APIView):
    def get(self, request):
        task_id = request.query_params.get("task_id")
        if not task_id:
            return Response({"message": "C2H5OH App is running!"}, status=status.HTTP_200_OK)
        result = AsyncResult(task_id)
        if result.state == "PENDING":
            return Response(
                {"task_id": task_id, "status": "pending"},
                status=status.HTTP_202_ACCEPTED
            )
        elif result.state == "STARTED":
            return Response(
                {"task_id": task_id, "status": "in progress"},
                status=status.HTTP_202_ACCEPTED
            )
        elif result.state == "SUCCESS":
            return Response(
                {
                    "task_id": task_id,
                    "status": "completed",
                    "result": result.result
                },
                status=status.HTTP_200_OK
            )
        elif result.state == "FAILURE":
            return Response(
                {
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(result.info)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        else:
            return Response(
                {"task_id": task_id, "status": result.state},
                status=status.HTTP_202_ACCEPTED
            )

    def _validate_file(self, file_obj):
        if not file_obj:
            raise ValueError("No file provided.")
        if not file_obj.name.endswith(".pkl"):
            raise ValueError("Invalid file type. Only .pkl files are allowed.")

    def post(self, request):
        file_obj = request.FILES.get("file")
        try:
            self._validate_file(file_obj)
            task_id = process_pickle_data(file_obj)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": "An unexpected error occurred."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(
            {"message": "Task started successfully!", "task_id": task_id},
            status=status.HTTP_202_ACCEPTED
        )
