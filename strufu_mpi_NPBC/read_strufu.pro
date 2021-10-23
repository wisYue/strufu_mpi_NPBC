pro read_strufu, filename, RET=ret, HEADER=header, NO_HEADER=NO_header

  ret = dblarr(10048,43)
  line     = dblarr(43)
  openr, 1, filename, ERROR = file_error
  if (file_error ne 0) then begin
     print, !ERR_STRING & close, 1 & stop
  endif
  print,'Reading file: '+filename
  header = ''
  if not keyword_set(no_header) then readf, 1, format='(A)', header
  line_number = 0l
  while not eof(1) do begin
     readf, 1, format='(43E30.8)', line
     ret[line_number,*] = line
     line_number = line_number + 1
  endwhile
  close, 1
  ret = ret[0:line_number-1,*]
  return

end
