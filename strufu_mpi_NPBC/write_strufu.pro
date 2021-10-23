pro write_strufu, filename, DAT=dat, HEADER=header

  openw, 1, filename, ERROR = file_error
  if (file_error ne 0) then begin
     print, !ERR_STRING & close, 1 & stop
  endif
  if keyword_set(header) then printf, 1, format='(A)', header
  for i = 0l, n_elements(dat[*,0])-1 do begin
     printf, 1, format='(43E-30.8)', dat[i,*]
  endfor
  close, 1
  print,'file written: '+filename
  return

end
