#!/usr/bin/perl
#
# Plot misfits of trial hypocentres in the NA search.
#
# Istvan Bondar
# International Seismological Centre
# August 2012
#
# Dependencies:
#    GMT4.5 or later.
#
# Usage:
#   gridsearch_plot.pl filename [gtlat gtlon gtot gtdepth [lat lon ot depth]]
# where
#   filename is a file written by isloc when the command line instruction
#      write_gridsearch_results=1 was given. The filename is constructed
#      by iscloc as
#      {isc_evid}.{locator_option}.gsres where locator option is
#         option = 0 free depth
#         option = 1 fix to region-dependent default depth
#         option = 2 fix depth to value provided by analyst
#         option = 3 fix depth to depth-phase depth
#         option = 4 fix location
#         option = 5 fix depth and location.
#   gtlat gtlon gtot gtdepth are the optional GT coordinates of the event
#      (latitude, longitude, origin epoch(!) time, and depth).
#   lat lon ot depth are the optional (final) coordinates of the event
#      (latitude, longitude, origin epoch(!) time, and depth).
# Output:
#   {isc_evid}.{locator_option}.gsres.ps
#
#
$|=1;
use strict;
use POSIX;
use Math::Trig qw(deg2rad);


my ($lat, $lon, $depth, $ot, $otini, $misfit, $i, $j, $s);
my ($r, $bx, $psfil, $pipe, $kmbar, $xtic, $ytic, $lo, $hi);
my ($epifix, $otfix, $depfix, $distol, $ottol, $deptol);
my ($fname, $gtlat, $gtlon, $gtot, $gtdepth, $isgt);
my ($flat, $flon, $fot, $fdepth, $isloc);
my $cpt = "misfit.cpt";
#
# check arguments, provide usage info and terminate if there are
# incorrect number of arguments
#
if ( @ARGV < 1 ) {
   print "Usage: gridsearch_plot.pl filename [gtlat gtlon gtot gtdepth ";
   print "[lat lon ot depth]]\n";
   exit -1;
}
$fname = $ARGV[0];
if (@ARGV > 1) {
    $gtlat = $ARGV[1];
    $gtlon = $ARGV[2];
    $gtot = $ARGV[3];
    $gtdepth = $ARGV[4];
    $isgt = 1;
}
if (@ARGV > 5) {
    $flat = $ARGV[5];
    $flon = $ARGV[6];
    $fot = $ARGV[7];
    $fdepth = $ARGV[8];
    $isloc = 1;
}
#
# set GMT parameters
#
`gmtset MEASURE_UNIT inch PLOT_DEGREE_FORMAT D`;
`gmtset PAGE_ORIENTATION landscape X_ORIGIN 1.5 Y_ORIGIN 2.5`;
`gmtset ANNOT_FONT_PRIMARY Helvetica-Bold ANNOT_FONT_SIZE_PRIMARY 14p`;
`gmtset ANNOT_FONT_SECONDARY Helvetica-Bold ANNOT_FONT_SIZE_SECONDARY 16p`;
`gmtset HEADER_FONT Helvetica-Bold HEADER_FONT_SIZE 18p`;
`gmtset LABEL_FONT Helvetica-Bold LABEL_FONT_SIZE 16p`;
#
# input and output files
#
$psfil = "$fname.ps";
open (IN, $fname) || die "ERROR : Cannot open $fname!\n";
#
# read input file header: ranges and initial hypocentre
#
(undef, undef, undef, $epifix, undef, undef, $distol) = split ' ', <IN>;
(undef, undef, undef, $otfix, undef, undef, $ottol) = split ' ', <IN>;
(undef, undef, undef, $depfix, undef, undef, $deptol) = split ' ', <IN>;
(undef, $lat, $lon, $otini, $depth) = split ' ', <IN>;
#
# print GT coordinates, if any
#
printf "GT: $gtlat $gtlon %f $gtdepth\n", $gtot - $otini if ($isgt);
printf "Final: $flat $flon %f $fdepth\n", $fot - $otini if ($isloc);
#
#
# plot epicenters
#
#
($r, $kmbar, $xtic, $ytic, undef) = projbox($distol, 4.2, $lat, $lon);
#($r, $kmbar, $xtic, $ytic, undef) = projbox(0.4, 4.2, $lat, $lon);
$bx = "-Ba$xtic"."g$xtic/a$ytic"."g$ytic:.$fname:WeSn";
`pscoast $r $bx $kmbar -Givory1 -Sazure1 -Wthin -A1000 -N1 -Di -K > $psfil`;
$pipe = "psxy $r -Sc0.02 -C$cpt -O -K >> $psfil";
open (PIPE, "| $pipe") || die "ERROR : Cannot open pipe!\n";
while (<IN>) {
    next if (/^#/);
    ($j, $i, $lat, $lon, $ot, $depth, $misfit) = split;
    next if ($i == -1);
    print PIPE "$lon $lat $misfit\n";
}
close PIPE;
close IN;
#
# plot best solution from grid search
#
`psxy $r -Ss0.1 -L -Wthin -O -K << END >> $psfil\n$lon $lat\nEND\n`;
#
# plot GT (if any)
#
$s = sprintf "%f %f", $gtlon, $gtlat;
`psxy $r -Sa0.1 -L -Wthin,magenta -O -K << END >> $psfil\n$s\nEND\n` if ($isgt);
#
# plot final location (if any)
#
$s = sprintf "%f %f", $flon, $flat;
`psxy $r -Sa0.1 -L -Wthin -O -K << END >> $psfil\n$s\nEND\n` if ($isloc);
#
# plot misfit scale
#
`psscale -C$cpt -D4.5/-0.8/9/0.12h -B1:Misfit: -O -K >> $psfil`;
#
#
# plot origin time deviations and depths
#
#
open (IN, $fname) || die "ERROR : Cannot open $fname!\n";
$lo = (($depth - $deptol) > 0.) ? $depth - $deptol : 0;
$hi = (($depth + $deptol) < 700.) ? $depth + $deptol : 700;
$r = sprintf "-R%.3f/%.3f/%.3f/%.3f -JX4.2/-4.2", -$ottol, $ottol, $lo, $hi;
(undef, undef, $xtic) = gettic(-$ottol, $ottol);
(undef, undef, $ytic) = gettic(-$lo, $hi);
$i = $xtic / 2;
$j = $ytic / 2;
$bx = "-Ba$xtic"."g$i:\"\@~D\@~OT [s]\":/a$ytic"."g$j:\"Depth [km]\":wESn";
`psbasemap $r $bx -X4.8 -O -K >> $psfil`;
$pipe = "psxy $r -Sc0.02 -C$cpt -O -K >> $psfil";
open (PIPE, "| $pipe") || die "ERROR : Cannot open pipe!\n";
while (<IN>) {
    next if (/^#/);
    ($j, $i, $lat, $lon, $ot, $depth, $misfit) = split;
    printf PIPE "%f %f %f\n", $ot - $otini, $depth, $misfit if ($i != -1);
#
#   print best solutions after each iteration in NA
#
    printf "$j $i $lat $lon %f $depth $misfit\n", $ot - $otini if ($i == -1);
}
close PIPE;
close IN;
#
# plot GT (if any)
#
$s = sprintf "%f %f", $gtot - $otini, $gtdepth if ($isgt);
`psxy $r -Sa0.1 -L -Wthin,magenta -O -K << END >> $psfil\n$s\nEND\n` if ($isgt);
#
# plot final location (if any)
#
$s = sprintf "%f %f", $fot - $otini, $fdepth if ($isloc);
`psxy $r -Sa0.1 -L -Wthin -O -K << END >> $psfil\n$s\nEND\n` if ($isloc);
#
# plot best solution from grid search
#
$s = sprintf "%f %f", $ot - $otini, $depth;
`psxy $r -Ss0.1 -L -Wthin -O << END >> $psfil\n$s\nEND\n`;

print "done.\n";

#
# EOF

#
#
# gettic
#   Returns tic interval and adjusted limits for an input range. The input
#   limits are adjusted so that the output limits are nice round numbers
#   and they can be divided into three or five round intervals.
#   Input:  input limits (min, max)
#   Output: output limits (min, max) and tic interval for annotation
#   Usage:  ($xlo, $xhi, $xtic) = gettic($xmin, $xmax);
#
#
sub gettic {
    my ($xmin, $xmax) = @_;
    my ($x, $y, $z, $u, $tic, $lo, $hi);
    $x = ($xmax - $xmin) / 3;
    $y = pow(10.0, floor(log10($x)));
    $z = $x / $y;
    $u = 2;
    $u = 1 if ($z < 1.5);
    $u = 5 if ($z > 3.5);
    $tic = $u * $y;
    $lo = floor($xmin / $tic) * $tic;
    $hi = ceil($xmax / $tic) * $tic;
    return ($lo, $hi, $tic);
}


#
#
# projbox
#   Calculates equidistant projection box encompassing a circle with
#   a given radius around the center point.
#   Input:  radius around center point in degrees
#           size of plot in inches
#           center's latitude
#           center's longitude
#   Output: projection parameters in GMT "-R -J" format
#           km scale bar in GMT "-L" format
#           longitude, latitude annotation intervals
#           lower left and upper right coordinates of bounding box
#   Usage:
#           ($r, $kmbar, $xtic, $ytic, $west, $south, $east, $north) =
#                   projbox($radius, $isize, $clat, $clon);
#           `pscoast $r -Ba$xtic/a$ytic $kmbar -Givory -Wthin -Dc > $psfil`;
#
#
sub projbox {
    my ($radius, $isize, $midlat, $midlon) = @_;
    my ($scale, $kmbar, $barlength, $r, $xtic, $ytic);
    my ($top, $bot, $left, $right, $ds, $x, $y, $z, $ispole);
    my ($ll_x, $ll_y, $ur_x, $ur_y, $west, $south, $east, $north);
#
#   scale
#
    $scale = $radius * 11112000. / ($isize * 2.54);
#
#   compute rough number of degrees north/south, east/west at the current scale
#
    $ds = 2.54 * $scale / 11112000.;
    $top = $midlat + $isize * $ds;
    $bot = $midlat - $isize * $ds;
    if (abs($midlat) < 90.) {
        $left = $midlon - $isize * $ds / cos(deg2rad($midlat));
        $right = $midlon + $isize * $ds / cos(deg2rad($midlat));
        $ispole = 0;
    }
    else {
        $left = -180.;
        $right = 180.;
        $ispole = 1;
    }
#
#   sanity checks
#
    if ($right - $left > 360) {
        $left = $midlon - 180;
        $right = $midlon + 180;
    }
    $top = 90  if ($top > 90);
    $bot = -90 if ($bot < -90);
#
#   rough plot range  based on scale and plot size
#
    $r = sprintf "-R%.3f/%.3f/%.3f/%.3fr -Je%.3f/%.3f/1:%.0f",
    $left, $bot, $right, $top, $midlon, $midlat, $scale;
#
#   compute the inches of the midlon, midlat point
#
    chomp($z = `echo $midlon $midlat | mapproject $r -Di`);
    ($x, $y) = split ' ',$z;
#
#   use the xy-inches of the midpoint to inverse project the inches to
#   get a rectangular plot of a particular size
#
    $ll_x = $x - $isize;
    $ll_y = $y - $isize;
    $ur_x = $x + $isize;
    $ur_y = $y + $isize;
#
#   after normalizing to get the center point, compute the lower-left and
#   upper-right points, guarentees a nice square plot.
#
    chomp($z = `echo $ll_x $ll_y | mapproject $r -Di -I`);
    ($west, $south) = split ' ',$z;
    chomp($z = `echo $ur_x $ur_y | mapproject $r -Di -I`);
    ($east, $north) = split ' ',$z;

    $east += 360 if ($east < $west);
#
#   projection
#
    $r = sprintf "-R%.3f/%.3f/%.3f/%.3fr -JE%.3f/%.3f/%.2f",
    $west, $south, $east, $north, $midlon, $midlat, $isize;
#
#   gridline intervals
#
    $xtic = gettic($west, $east);
    if ($ispole) {
        $ytic = 5.;
    }
    else {
        $ytic = gettic($south, $north);
    }
#
#   scalebar
#
    $barlength = $isize * 0.5;
    $barlength *= 2.54 * $scale / 100000;
    $barlength = gettic(0., 3. * $barlength);
    $barlength += 1 if ($barlength < 1);
    $kmbar = sprintf "-Lfx%.3f/%.3f/%.3f/%d", $isize / 4, $isize / 8,
    $midlat, int($barlength);
#
#   return projection, kmbar flags, gridline intervals and coordinates
#   of lower left and upper right corners
#
    return ($r, $kmbar, $xtic, $ytic, $west, $south, $east, $north);
}

