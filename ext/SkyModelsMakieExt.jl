module SkyModelsMakieExt



using SkyModels
if isdefined(Base, :get_extension)
    using Makie
else
    using ..Makie
end

function Makie.convert_arguments(::Makie.SurfaceLike, img::IntensityMap{T,2}) where {T}
    (;X, Y) = img
    Xr = rad2μas.(X)
    Yr = rad2μas.(Y)
    return Xr, Yr, ComradeBase.baseimage(img)
end



end
