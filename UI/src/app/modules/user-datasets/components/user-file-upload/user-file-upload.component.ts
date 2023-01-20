import {Component, OnInit} from '@angular/core';
import {FormBuilder, Validators} from '@angular/forms';
import {UserDatasetsService} from '../../services/user-datasets.service';
import {ToastService} from '../../../core/services/toast/toast.service';
import {BehaviorSubject, Subscription} from 'rxjs';
import {finalize} from 'rxjs/operators';
import {HttpEventType} from '@angular/common/http';
import {allowedNameValidator} from '../../services/allowed-name.directive';
import * as uuid from 'uuid';


@Component({
  selector: 'qa-user-file-upload',
  templateUrl: './user-file-upload.component.html',
  styleUrls: ['./user-file-upload.component.scss']
})
export class UserFileUploadComponent implements OnInit {
  // variables to store file information
  file: File;
  fileName = '';
  name = '';
  // variable to open the form
  dialogVisible = false;
  spinnerVisible = false;

  uploadProgress: BehaviorSubject<number> = new BehaviorSubject<number>(0);
  uploadSub: Subscription;

  // dataset file form
  metadataForm = this.formBuilder.group({
    dataset_name: [null, [Validators.required, Validators.maxLength(30), allowedNameValidator()]],
    dataset_pretty_name: [null, [Validators.maxLength(30), allowedNameValidator(true)]],
    version_name: [null, [Validators.required, Validators.maxLength(30), allowedNameValidator()]],
    version_pretty_name: [null, [Validators.maxLength(30), allowedNameValidator(true)]],
  });

  constructor(private userDatasetService: UserDatasetsService,
              private formBuilder: FormBuilder,
              private toastService: ToastService) {
  }

  ngOnInit(): void {
  }

  onFileSelected(event): void {
    this.file = event.target.files[0];
    const fileExtension =  this.file.name.split('.').reverse()[0];
    this.fileName = `${uuid.v4()}.${fileExtension}`;
    this.dialogVisible = true;
    // I need to clean the selected file, otherwise there will be problem with choosing the same file next time
    event.target.value = null;
  }

  sendForm(): void {
    if (this.file) {
      this.name = 'uploadedFile';
      this.spinnerVisible = true;
      const upload$ = this.userDatasetService.userFileUpload(this.name, this.file, this.fileName)
        .pipe(finalize(() => this.reset));

      this.uploadSub = upload$.subscribe(event => {
        if (event.type === HttpEventType.UploadProgress) {
          this.uploadProgress.next(Math.round(100 * (event.loaded / event.total)));
        } else if (event.type === HttpEventType.Response) {
          this.userDatasetService.sendMetadata(this.metadataForm.value, event.body.id).subscribe(() => {
              this.userDatasetService.refresh.next(true);
              this.resetFile();
            },
            (message) => {
              this.spinnerVisible = false;
              this.toastService.showErrorWithHeader('Metadata not saved.',
                `${message.error.error}.\n Provided metadata could not be saved. Please try again or contact our team.`);
            },
            () => {
              this.spinnerVisible = false;
              this.metadataForm.reset('');
            });
        }
      },
        (message) => {
          this.spinnerVisible = false;
          console.log(message);
          this.toastService.showErrorWithHeader('File not saved',
            `${message.error.error}.\n File could not be uploaded. Please try again or contact our team.`);
        }
        );
    }
  }

  onSaveData(): void {
    this.dialogVisible = false;
  }

  resetFile(): void{
    this.file = null;
    this.fileName = null;
  }

  reset(): void {
    this.uploadProgress = null;
    this.uploadSub = null;
  }

}